# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.network
================

EpistemicGraph — orchestrates multiple EpistemicState nodes as a directed,
influence-aware graph.

Why this version
----------------
- Deep telemetry: every applied edge influence is logged on the destination node
  (`event_log`), plus per-edge counters for diagnostics and dashboards.
- Determinism hooks: optional RNG to coordinate any stochastic decisions, aligned
  with the seeded RNG style used in `EpistemicState` / `PolicyManager`.
- Safer vector nudging: prefers a node's own current direction before falling
  back to source direction and, only last, a ones vector.
- Bounded & reversible policy nudges: transient stabilization via Θ↑ and k↓,
  logged and capped; no permanent policy mutation here (aligns with meta_learning).
- Utilities that align with the rest of the codebase: adjacency export/import,
  cascade traces, graph-level stats, and a propagation suspend context.
- Graph-level hooks, validation, replay/simulation helpers, and edge pruning
  for production operations.

Public API (compatible + extras)
--------------------------------
- class EpistemicGraph
  - add(name, state=None, **kwargs)
  - link(src, dst, weight=0.5, mode="pressure", decay=0.85, cooldown=5)
  - update_link(src, dst, **updates)
  - unlink(src, dst)
  - neighbors(src)
  - step(name, R)
  - step_all({name: R, ...})
  - broadcast(R, nodes=None)
  - suspend_propagation()   # context manager
  - stats() / diagnostics()
  - adjacency() / save_graph(path) / load_graph(path)
  - last_cascade(k=20) / clear_cascade()
  - validate(allow_self_loops=False)
  - degree(name=None)
  - reset_counters()
  - top_edges(by="applied_sum", k=10)
  - link_from_matrix(names, W, mode, decay, cooldown)
  - replay(sequence_of_evidence_dicts)
  - predict_influence(src, dst, post=None, rupture=None, depth=1)
  - register_hook(event, fn)   # events: on_influence, on_link, on_unlink

Notes
-----
- Modes: "pressure" propagates rupture pressure; "delta" gives weak coupling
  without rupture; "policy" gently biases Θ and k under stress.
- Works with scalar or vector V. Steps are bounded by min(graph.max_step,
  node.step_cap) and damped by an oscillation guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, List, Any, Tuple
import json
import numpy as np

from .epistemic import EpistemicState

# ---------------------------
# Edge definition
# ---------------------------

@dataclass
class Edge:
    """Directed influence from src -> dst.

    Attributes
    ----------
    weight : float
        Base multiplier on influence.
    mode : str
        "pressure" | "delta" | "policy".
    decay : float
        Per-hop attenuation when cascading.
    cooldown : int
        Minimum destination steps between consecutive influences on this edge.
    last_influence_t : int
        Destination node's last step index when this edge fired.
    hits : int
        Number of times this edge applied a non-zero influence.
    applied_sum : float
        Sum of applied magnitudes (for coarse heatmaps).
    applied_ema : float
        Exponential moving average of |applied| (smooth heatmap / alerting).
    ema_alpha : float
        EMA smoothing factor in (0,1].
    """
    weight: float = 0.5
    mode: str = "pressure"
    decay: float = 0.85
    cooldown: int = 5
    last_influence_t: int = -999_999
    hits: int = 0
    applied_sum: float = 0.0
    applied_ema: float = 0.0
    ema_alpha: float = 0.2

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weight": float(self.weight),
            "mode": self.mode,
            "decay": float(self.decay),
            "cooldown": int(self.cooldown),
            "last_influence_t": int(self.last_influence_t),
            "hits": int(self.hits),
            "applied_sum": float(self.applied_sum),
            "applied_ema": float(self.applied_ema),
            "ema_alpha": float(self.ema_alpha),
        }


# ---------------------------
# Graph
# ---------------------------

class EpistemicGraph:
    """
    Orchestrates multiple `EpistemicState` nodes with directional influence links.

    Parameters
    ----------
    damping : float
        Global damping multiplier on all propagated influences. Default 0.5.
    max_depth : int
        Maximum cascade depth when propagating influence. Default 3.
    max_step : float
        Global cap on a single propagated nudge magnitude. Default 1.0.
    rupture_only_propagation : bool
        If True, only propagate from a node when it ruptures, except that
        "delta" edges still couple (weakly) without rupture. Default True.
    rng : np.random.Generator | None
        Optional RNG to align with seeded determinism across the system.
    cascade_trace_cap : int
        Keep the last N influence records for quick debugging (default 512).
    """

    # ---- construction ----
    def __init__(
        self,
        damping: float = 0.5,
        max_depth: int = 3,
        max_step: float = 1.0,
        rupture_only_propagation: bool = True,
        rng: Optional[np.random.Generator] = None,
        cascade_trace_cap: int = 512,
    ):
        self.nodes: Dict[str, EpistemicState] = {}
        self.edges: Dict[str, Dict[str, Edge]] = {}
        self.damping = float(damping)
        self.max_depth = int(max_depth)
        self.max_step = float(max_step)
        self.rupture_only = bool(rupture_only_propagation)
        self.graph_rng: np.random.Generator = rng or np.random.default_rng()

        # Lightweight in-memory trace of last influences for UIs / debugging
        self._cascade_trace: List[Dict[str, Any]] = []
        self._cascade_trace_cap = int(max(0, cascade_trace_cap))

        # Global switch to silence propagation (e.g., during batch updates)
        self._suspended: bool = False

        # Graph-level hooks: event -> list[Callable[[dict], None]]
        self._hooks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

        # Graph event log (lightweight)
        self.event_log: List[Dict[str, Any]] = []
        self._t: int = 0  # coarse graph time (increments on each step())

    # ---- node & edge management ----

    def add(self, name: str, state: Optional[EpistemicState] = None, **kwargs) -> None:
        """Add a node.

        Parameters
        ----------
        name : str
            Node name (must be unique).
        state : EpistemicState, optional
            An existing state to insert. If None, a new one is constructed from kwargs.
        **kwargs :
            Passed to `EpistemicState(**kwargs)` when `state` is None.
        """
        if name in self.nodes:
            raise KeyError(f"Node '{name}' already exists")
        if state is None:
            state = EpistemicState(**kwargs)
        self.nodes[name] = state
        self.edges.setdefault(name, {})
        # Telemetry on the node itself
        try:
            state._log_event("graph_node_added", {"node": name})
        except Exception:
            pass
        self._emit("on_link", {"event": "node_added", "node": name})

    def link(
        self,
        src: str,
        dst: str,
        weight: float = 0.5,
        mode: str = "pressure",
        decay: float = 0.85,
        cooldown: int = 5,
    ) -> None:
        """Create/overwrite a directed link `src -> dst`.

        mode ∈ {"pressure", "delta", "policy"}
        """
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist in graph.")
        if mode not in ("pressure", "delta", "policy"):
            raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
        self.edges[src][dst] = Edge(
            weight=float(weight), mode=mode, decay=float(decay), cooldown=int(cooldown)
        )
        # Telemetry on both endpoints
        try:
            self.nodes[src]._log_event("graph_edge_linked", {"src": src, "dst": dst, "mode": mode, "weight": float(weight), "decay": float(decay), "cooldown": int(cooldown)})
            self.nodes[dst]._log_event("graph_edge_linked", {"src": src, "dst": dst, "mode": mode, "weight": float(weight), "decay": float(decay), "cooldown": int(cooldown)})
        except Exception:
            pass
        self._emit("on_link", {"event": "edge_linked", "src": src, "dst": dst, "mode": mode})

    def update_link(self, src: str, dst: str, **updates: Any) -> None:
        """Update parameters of an existing link in-place (weight/decay/cooldown/mode)."""
        e = self.edges.get(src, {}).get(dst)
        if not e:
            raise KeyError(f"No edge {src}->{dst}")
        for k in ("weight", "decay", "cooldown", "mode"):
            if k in updates:
                setattr(e, k, updates[k] if k != "mode" else str(updates[k]))
        try:
            self.nodes[dst]._log_event("graph_edge_updated", {"src": src, "dst": dst, **{k: updates[k] for k in updates if k in {"weight", "decay", "cooldown", "mode"}}})
        except Exception:
            pass
        self._emit("on_link", {"event": "edge_updated", "src": src, "dst": dst, **updates})

    def unlink(self, src: str, dst: str) -> None:
        """Remove the directed link `src -> dst` if it exists."""
        if src in self.edges and dst in self.edges[src]:
            self.edges[src].pop(dst, None)
            try:
                self.nodes[dst]._log_event("graph_edge_unlinked", {"src": src, "dst": dst})
            except Exception:
                pass
            self._emit("on_unlink", {"event": "edge_unlinked", "src": src, "dst": dst})

    def neighbors(self, src: str) -> Dict[str, Edge]:
        """Return the adjacency map for `src`."""
        return self.edges.get(src, {})

    # ---- stepping ----

    def step(self, name: str, R: Any) -> Dict[str, Any]:
        """Feed evidence to a node and (optionally) propagate influence.

        Returns the node's last log entry (convenience for dashboards).
        """
        if name not in self.nodes:
            raise KeyError(f"Unknown node '{name}'")
        n = self.nodes[name]

        pre_ruptured = bool(n.last().get("ruptured")) if n.last() else False

        n.receive(R, source=name)
        post = n.last() or {}
        ruptured = bool(post.get("ruptured", False))

        try:
            n._log_event("graph_step", {"node": name, "ruptured": ruptured, "∆": float(post.get("∆", 0.0)), "Θ": float(post.get("Θ", 0.0))})
        except Exception:
            pass

        if not self._suspended:
            if self.rupture_only and not ruptured:
                # Allow continuous coupling only for "delta" edges
                self._propagate_from(name, depth=1, rupture=False)
            else:
                self._propagate_from(name, depth=1, rupture=ruptured)

        post["_osc_note"] = "flip" if ruptured != pre_ruptured else "steady"
        self._t += 1
        return post

    def step_all(self, evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Batch step: process `{node_name: R}` for each item in `evidence`.
        Nodes without evidence are skipped.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, R in evidence.items():
            out[name] = self.step(name, R)
        return out

    def broadcast(self, R: Any, nodes: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Feed the same evidence to many nodes (e.g., common shocks)."""
        targets = list(nodes) if nodes else list(self.nodes.keys())
        return {name: self.step(name, R) for name in targets}

    # ---- suspend / resume ----

    def suspend_propagation(self) -> "_SuspendCtx":
        """Context manager to temporarily disable graph propagation."""
        return _SuspendCtx(self)

    # ---- propagation core ----

    def _propagate_from(self, src: str, depth: int, rupture: bool) -> None:
        """Recursively propagate influence from `src` up to `max_depth`."""
        if depth > self.max_depth:
            return
        post = self.nodes[src].last() or {}
        delta = float(post.get("∆", 0.0))
        theta = float(post.get("Θ", 0.0))
        pressure = max(0.0, delta - theta)  # rupture pressure; zero if no rupture

        for dst, e in self.neighbors(src).items():
            dst_state = self.nodes[dst]

            # Edge cooldown check (relative to destination node's own step counter)
            if (dst_state.summary()["t"] - e.last_influence_t) < e.cooldown:
                continue

            # Base influence magnitude by mode
            if e.mode == "pressure":
                if not rupture:
                    continue
                base = pressure
            elif e.mode == "delta":
                base = delta * 0.25  # continuous, weaker coupling
            else:  # "policy"
                base = pressure if rupture else 0.0

            if base <= 0.0:
                continue

            magnitude = float(
                self.damping * e.weight * base * (e.decay ** (depth - 1))
            )

            # Cap by both graph-level and node-local caps
            node_cap = float(getattr(dst_state, "step_cap", self.max_step))
            cap = float(min(self.max_step, max(1e-6, node_cap)))
            magnitude = float(np.clip(magnitude, -cap, cap))

            # Oscillation guard on destination
            magnitude *= self._oscillation_factor(dst_state)

            # Apply influence
            applied = 0.0
            if e.mode in ("pressure", "delta"):
                applied = self._nudge_value_toward_recent_R(dst_state, magnitude, src_post=post)
            elif e.mode == "policy" and magnitude > 0.0:
                applied = self._nudge_policy(dst_state, magnitude)

            if applied != 0.0:
                e.hits += 1
                e.applied_sum += float(abs(applied))
                e.applied_ema = float(e.ema_alpha * abs(applied) + (1.0 - e.ema_alpha) * e.applied_ema)

            # Log influence for traceability
            event = {
                "t": self._t,
                "src": src,
                "dst": dst,
                "mode": e.mode,
                "depth": depth,
                "base": base,
                "magnitude_capped": float(magnitude),
                "applied": float(applied),
            }
            self._record_cascade(event)
            try:
                dst_state._log_event("graph_influence", event)
            except Exception:
                pass  # keep network resilient
            self._emit("on_influence", event)

            # Mark last influence step for cooldown
            e.last_influence_t = dst_state.summary()["t"]

            # Recurse
            self._propagate_from(dst, depth + 1, rupture=rupture)

    # ---- influence primitives ----

    @staticmethod
    def _last_R_scalar_or_vec(state: EpistemicState) -> Optional[np.ndarray]:
        """Fetch last seen R as a vector if possible; fallback to 1-D vector from scalar."""
        last = state.last()
        if not last:
            return None
        R = last.get("R")
        if R is None:
            return None
        if isinstance(R, (list, np.ndarray)):
            return np.asarray(R, dtype=float)
        try:
            return np.array([float(R)], dtype=float)
        except Exception:
            return None

    def _nudge_value_toward_recent_R(
        self,
        dst_state: EpistemicState,
        magnitude: float,
        src_post: Dict[str, Any],
    ) -> float:
        """Move `dst_state.V` a little toward its own last `R` (preferred) or toward
        the source node’s `R` direction as a fallback. Works for scalars and vectors.
        Returns the L2 norm of the applied step (for logging)."""
        dst_R_vec = self._last_R_scalar_or_vec(dst_state)

        if isinstance(dst_state.V, np.ndarray):
            v = np.asarray(dst_state.V, dtype=float)
            n_v = float(np.linalg.norm(v))

            if dst_R_vec is not None:
                direction = dst_R_vec / (np.linalg.norm(dst_R_vec) or 1.0)
            else:
                # Prefer self-direction if available; last resort: source R direction; then ones
                direction: Optional[np.ndarray]
                direction = (v / (n_v or 1.0)) if n_v > 0 else None
                if direction is None:
                    src_R = src_post.get("R")
                    if isinstance(src_R, (list, np.ndarray)):
                        rvec = np.asarray(src_R, dtype=float)
                        direction = rvec / (np.linalg.norm(rvec) or 1.0)
                    else:
                        direction = np.ones_like(v)

            step = direction * float(magnitude)

            # Respect destination node’s cap as well
            node_cap = float(getattr(dst_state, "step_cap", self.max_step))
            cap = float(min(self.max_step, max(1e-6, node_cap)))
            step_norm = float(np.linalg.norm(step))
            if step_norm > cap:
                step = step * (cap / (step_norm or 1.0))
                step_norm = cap

            dst_state.V = (v + step).astype(float)
            return step_norm

        else:
            # Scalar destination
            if dst_R_vec is not None:
                target = float(dst_R_vec.mean())
            else:
                src_R = src_post.get("R")
                try:
                    target = float(
                        src_R if not isinstance(src_R, (list, np.ndarray)) else np.linalg.norm(src_R)
                    )
                except Exception:
                    target = float(dst_state.V)

            sign = 1.0 if (target - float(dst_state.V)) >= 0.0 else -1.0
            node_cap = float(getattr(dst_state, "step_cap", self.max_step))
            cap = float(min(self.max_step, max(1e-6, node_cap)))
            step = sign * float(np.clip(magnitude, -cap, cap))
            dst_state.V = float(dst_state.V) + step
            return abs(step)

    @staticmethod
    def _nudge_policy(dst_state: EpistemicState, magnitude: float) -> float:
        """Light-touch runtime bias: slightly increase Θ and reduce k under stress.
        Reversible; meant for transient stabilization, not persistent mutation.
        Returns the scalar 'policy adjustment magnitude' applied (for logging)."""
        try:
            dθ = float(np.clip(0.05 * magnitude, -0.2, 0.2))
            dk = float(np.clip(0.03 * magnitude, 0.0, 0.2))
            dst_state.Θ = max(1e-6, float(dst_state.Θ) + dθ)
            dst_state.k = max(1e-3, float(dst_state.k) - dk)
            return abs(dθ) + abs(dk)
        except Exception:
            return 0.0  # keep network resilient to odd node states

    # ---- diagnostics ----

    @staticmethod
    def _oscillation_factor(state: EpistemicState, window: int = 20) -> float:
        """Compute a damping multiplier in [0.5, 1.0] based on rupture flip frequency.
        More flips -> smaller factor."""
        hist = state.history[-window:]
        if len(hist) < 4:
            return 1.0
        rupt = np.array([1 if h.get("ruptured", False) else 0 for h in hist], dtype=int)
        flips = np.abs(np.diff(rupt)).sum()
        factor = 1.0 - min(0.5, flips / max(8, window))
        return float(np.clip(factor, 0.5, 1.0))

    def stats(self) -> Dict[str, Any]:
        """Quick aggregate snapshot for dashboards.

        Returns
        -------
        {node_name: {"ruptures", "mean_drift", "std_drift", "last_symbol"}}
        """
        out: Dict[str, Any] = {}
        for name, s in self.nodes.items():
            ds = s.drift_stats(window=min(50, len(s.history)))
            out[name] = {
                "ruptures": s.summary()["ruptures"],
                "mean_drift": ds.get("mean_drift", 0.0),
                "std_drift": ds.get("std_drift", 0.0),
                "last_symbol": s.symbol(),
            }
        return out

    def diagnostics(self) -> Dict[str, Any]:
        """Detailed graph diagnostics: node stats + edge counters."""
        node_stats = self.stats()
        edge_stats: Dict[str, Dict[str, Any]] = {}
        for src, nbrs in self.edges.items():
            for dst, e in nbrs.items():
                edge_stats[f"{src}->{dst}"] = e.as_dict()
        return {"nodes": node_stats, "edges": edge_stats}

    def adjacency(self) -> Dict[str, Dict[str, dict]]:
        """Return the adjacency with edge metadata for visualization.

        Returns
        -------
        {src: {dst: {"weight", "mode", "decay", "cooldown"}}}
        """
        adj: Dict[str, Dict[str, dict]] = {}
        for src, nbrs in self.edges.items():
            adj[src] = {
                dst: {
                    "weight": float(e.weight),
                    "mode": e.mode,
                    "decay": float(e.decay),
                    "cooldown": int(e.cooldown),
                }
                for dst, e in nbrs.items()
            }
        return adj

    # ---- I/O: save/load adjacency ----

    def save_graph(self, path: str) -> None:
        """Persist only the adjacency/edge metadata to JSON (not node internals)."""
        payload = {
            "damping": self.damping,
            "max_depth": self.max_depth,
            "max_step": self.max_step,
            "rupture_only": self.rupture_only,
            "nodes": list(self.nodes.keys()),
            "edges": {
                src: {dst: e.as_dict() for dst, e in nbrs.items()}
                for src, nbrs in self.edges.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_graph(self, path: str) -> None:
        """Load adjacency/edge metadata from JSON. Nodes must pre-exist or will be created with defaults."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.damping = float(payload.get("damping", self.damping))
        self.max_depth = int(payload.get("max_depth", self.max_depth))
        self.max_step = float(payload.get("max_step", self.max_step))
        self.rupture_only = bool(payload.get("rupture_only", self.rupture_only))

        # Ensure nodes exist
        for name in payload.get("nodes", []):
            if name not in self.nodes:
                self.add(name)

        # Rebuild edges
        self.edges = {src: {} for src in self.nodes.keys()}
        for src, nbrs in payload.get("edges", {}).items():
            if src not in self.nodes:
                continue
            for dst, meta in nbrs.items():
                if dst not in self.nodes:
                    self.add(dst)
                e = Edge(
                    weight=float(meta.get("weight", 0.5)),
                    mode=str(meta.get("mode", "pressure")),
                    decay=float(meta.get("decay", 0.85)),
                    cooldown=int(meta.get("cooldown", 5)),
                )
                e.last_influence_t = int(meta.get("last_influence_t", -999_999))
                e.hits = int(meta.get("hits", 0))
                e.applied_sum = float(meta.get("applied_sum", 0.0))
                e.applied_ema = float(meta.get("applied_ema", 0.0))
                e.ema_alpha = float(meta.get("ema_alpha", 0.2))
                self.edges[src][dst] = e

    # ---- cascade trace ----

    def _record_cascade(self, event: Dict[str, Any]) -> None:
        if self._cascade_trace_cap <= 0:
            return
        self._cascade_trace.append(event)
        if len(self._cascade_trace) > self._cascade_trace_cap:
            self._cascade_trace = self._cascade_trace[-self._cascade_trace_cap:]
        # mirror into graph event log
        self.event_log.append({"event": "influence", **event})

    def last_cascade(self, k: int = 20) -> List[Dict[str, Any]]:
        """Return last k influence events from the rolling cascade trace."""
        if k <= 0:
            return []
        return self._cascade_trace[-int(k):]

    def clear_cascade(self) -> None:
        """Clear the in-memory cascade trace buffer."""
        self._cascade_trace.clear()

    # ---- utilities / operations ----

    def validate(self, allow_self_loops: bool = False) -> Dict[str, Any]:
        """Validate topology and return a report."""
        issues: List[str] = []
        for src, nbrs in self.edges.items():
            for dst, e in nbrs.items():
                if (not allow_self_loops) and src == dst:
                    issues.append(f"self-loop {src}->{dst}")
                if e.decay <= 0 or e.decay > 1.0:
                    issues.append(f"decay out of (0,1]: {src}->{dst} decay={e.decay}")
                if e.cooldown < 0:
                    issues.append(f"negative cooldown on {src}->{dst}")
        return {"ok": len(issues) == 0, "issues": issues}

    def degree(self, name: Optional[str] = None) -> Any:
        """Return (in_degree, out_degree) for a node, or dict for all."""
        if name is None:
            out: Dict[str, Tuple[int, int]] = {}
            for n in self.nodes:
                out[n] = (sum(1 for src in self.edges if n in self.edges[src]), len(self.edges.get(n, {})))
            return out
        return (sum(1 for src in self.edges if name in self.edges[src]), len(self.edges.get(name, {})))

    def reset_counters(self) -> None:
        """Reset per-edge counters (hits/applied_*)."""
        for src, nbrs in self.edges.items():
            for e in nbrs.values():
                e.hits = 0
                e.applied_sum = 0.0
                e.applied_ema = 0.0

    def top_edges(self, by: str = "applied_sum", k: int = 10) -> List[Tuple[str, str, float]]:
        """Return top-k edges by a metric (applied_sum|applied_ema|hits)."""
        metrics: List[Tuple[str, str, float]] = []
        for src, nbrs in self.edges.items():
            for dst, e in nbrs.items():
                val = float(getattr(e, by, 0.0)) if by != "hits" else float(e.hits)
                metrics.append((src, dst, val))
        metrics.sort(key=lambda x: x[2], reverse=True)
        return metrics[:max(0, int(k))]

    def prune_edges(self, min_hits: Optional[int] = None, min_applied_sum: Optional[float] = None, min_ema: Optional[float] = None) -> int:
        """Remove edges below thresholds. Returns number of removed edges."""
        to_remove: List[Tuple[str, str]] = []
        for src, nbrs in self.edges.items():
            for dst, e in nbrs.items():
                if (min_hits is not None and e.hits < min_hits) or \
                   (min_applied_sum is not None and e.applied_sum < min_applied_sum) or \
                   (min_ema is not None and e.applied_ema < min_ema):
                    to_remove.append((src, dst))
        for src, dst in to_remove:
            self.unlink(src, dst)
        return len(to_remove)

    def link_from_matrix(self, names: List[str], W: np.ndarray, mode: str = "pressure", decay: float = 0.85, cooldown: int = 5) -> None:
        """Create edges from a weight matrix W (shape n×n). Diagonal is ignored by default."""
        W = np.asarray(W, dtype=float)
        assert W.shape[0] == W.shape[1] == len(names), "W must be square and match names length"
        for i, src in enumerate(names):
            if src not in self.nodes:
                self.add(src)
        for j, dst in enumerate(names):
            if dst not in self.nodes:
                self.add(dst)
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j:
                    continue
                w = float(W[i, j])
                if w == 0.0:
                    continue
                self.link(src, dst, weight=abs(w), mode=mode, decay=decay, cooldown=cooldown)

    def replay(self, evidence_sequence: List[Dict[str, Any]]) -> List[Dict[str, Dict[str, Any]]]:
        """Replay a list of evidence dicts (each is {node: R}) with live propagation.
        Returns per-step snapshots mapping node->last_log.
        """
        out: List[Dict[str, Dict[str, Any]]] = []
        for step_ev in evidence_sequence:
            out.append(self.step_all(step_ev))
        return out

    def predict_influence(self, src: str, dst: str, post: Optional[Dict[str, Any]] = None, rupture: Optional[bool] = None, depth: int = 1) -> float:
        """Compute the capped magnitude that *would* be applied from src->dst for the given `post` (or last src post).
        Does not mutate state.
        """
        if src not in self.nodes or dst not in self.nodes:
            return 0.0
        if depth < 1:
            depth = 1
        src_post = post or (self.nodes[src].last() or {})
        delta = float(src_post.get("∆", 0.0))
        theta = float(src_post.get("Θ", 0.0))
        pressure = max(0.0, delta - theta)
        e = self.edges.get(src, {}).get(dst)
        if not e:
            return 0.0
        if rupture is None:
            rupture = bool(src_post.get("ruptured", False))
        if e.mode == "pressure" and not rupture:
            return 0.0
        base = (pressure if e.mode in ("pressure", "policy") else delta * 0.25)
        if base <= 0.0:
            return 0.0
        magnitude = float(self.damping * e.weight * base * (e.decay ** (depth - 1)))
        node_cap = float(getattr(self.nodes[dst], "step_cap", self.max_step))
        cap = float(min(self.max_step, max(1e-6, node_cap)))
        magnitude = float(np.clip(magnitude, -cap, cap))
        magnitude *= self._oscillation_factor(self.nodes[dst])
        return float(abs(magnitude))

    # ---- hooks ----

    def register_hook(self, event: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        if not isinstance(event, str):
            raise TypeError("event must be a string")
        if not callable(fn):
            raise TypeError("hook must be callable")
        self._hooks.setdefault(event, []).append(fn)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for fn in self._hooks.get(event, []):
            try:
                fn(dict(payload))
            except Exception:
                # Do not let user hooks destabilize the graph
                pass
        # also mirror to graph event log
        self.event_log.append({"event": event, **payload})


# ---------------------------
# Context manager
# ---------------------------

class _SuspendCtx:
    def __init__(self, g: EpistemicGraph):
        self._g = g
        self._prior = g._suspended

    def __enter__(self) -> "_SuspendCtx":
        self._prior = self._g._suspended
        self._g._suspended = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._g._suspended = self._prior
