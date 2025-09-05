# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.network
================

EpistemicGraph — orchestrates multiple EpistemicState nodes as a directed,
influence-aware graph.

Highlights
----------
- Deep telemetry: every applied edge influence is logged on the destination node
  (`event_log`) and mirrored into a rolling graph-level cascade trace.
- Determinism hooks: optional RNG to coordinate any stochastic decisions, aligned
  with the seeded RNG style used in `EpistemicState` / `PolicyManager`.
- Safer vector nudging: prefers a node’s own recent R-direction, then its V̂,
  then (normalized) ones; all nudges are step-capped.
- Bounded & reversible policy nudges: transient stabilization via Θ↑ and k↓ with
  decaying bias; logged and capped; no permanent policy mutation.
- Utilities: adjacency export/import (topology only), cascade traces, graph stats,
  suspend context, validation, replay, edge pruning.

Programmable layer
------------------
- Fully injectable strategies (gate/influence/magnitude/target/nudge/damp)
- Persistence by **reference**: strategy IDs + JSON-safe params are saved/loaded.
  Callables are looked up via a module-level registry; code is never serialized.

Public API
----------
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

- class EpistemicProgrammableGraph (subclass)
  - set_default(name, fn)       # override graph-wide strategy defaults
  - set_position(name, x, y)    # optional node positions for distance-aware decay
  - link(..., gate_fn=..., influence_fn=..., magnitude_fn=..., target_fn=...,
                 nudge_fn=..., damp_fn=..., target=..., resolve_channel=..., params={...},
                 strategy_id="name@semver")
  - update_link(..., gate_fn=..., influence_fn=..., ...)  # hot-update callables/params
  - save_graph(path, include_strategies=True)
  - load_graph(path, *, strict_strategies=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, List, Any, Tuple, Protocol, Union
import json
import math
import numpy as np

from .epistemic import EpistemicState

__all__ = [
    "Edge",
    "ProgrammableEdge",
    "EpistemicGraph",
    "EpistemicProgrammableGraph",
    "register_strategy",
    "get_strategy",
]

# =============================================================================
# Strategy registry (persist-by-reference for programmable graphs)
# =============================================================================

_STRATEGY_REGISTRY: Dict[str, Dict[str, Callable]] = {}

def register_strategy(strategy_id: str, **fns: Callable) -> None:
    """
    Register a strategy bundle by ID. Only known keys are accepted:
      gate_fn, influence_fn, magnitude_fn, target_fn, nudge_fn, damp_fn
    """
    allowed = {"gate_fn", "influence_fn", "magnitude_fn", "target_fn", "nudge_fn", "damp_fn"}
    unknown = set(fns) - allowed
    if unknown:
        raise KeyError(f"Unknown strategy keys: {unknown}")
    _STRATEGY_REGISTRY[strategy_id] = {k: v for k, v in fns.items() if v is not None}

def get_strategy(strategy_id: str) -> Optional[Dict[str, Callable]]:
    """Return strategy callables dict for an ID, or None if missing."""
    return _STRATEGY_REGISTRY.get(strategy_id)

# ---------------------------
# JSON safety helpers
# ---------------------------

def _to_py_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x

def _is_json_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _sanitize_float(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    return xf if math.isfinite(xf) else 0.0

def _json_sanitize(obj: Any, _depth: int = 0, _max_depth: int = 64) -> Any:
    """
    Convert arbitrary Python/numpy structures into JSON-safe values.
    - Converts numpy scalars/arrays to Python scalars/lists
    - Tuples/sets -> lists
    - Dict keys -> str
    - Callables -> None (dropped by callers as needed)
    - Non-finite floats -> 0.0
    """
    if _depth > _max_depth:
        return None

    # Strip callables early
    if callable(obj):
        return None

    # Numpy scalars
    if isinstance(obj, np.generic):
        obj = obj.item()

    # Scalars
    if _is_json_scalar(obj):
        if isinstance(obj, float):
            return _sanitize_float(obj)
        return obj

    # Numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(_to_py_scalar(v), _depth + 1, _max_depth) for v in obj.tolist()]

    # Lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v, _depth + 1, _max_depth) for v in obj]

    # Dicts
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if callable(v):
                continue
            ks = str(k)
            sv = _json_sanitize(v, _depth + 1, _max_depth)
            # allow only JSON-native types post-sanitization
            if isinstance(sv, (dict, list)) or _is_json_scalar(sv):
                out[ks] = sv
        return out

    # Fallback: try repr() (last resort)
    try:
        s = repr(obj)
        return s
    except Exception:
        return None

# ---------------------------
# Edge definition
# ---------------------------

@dataclass
class Edge:
    """Directed influence from src -> dst."""
    weight: float = 0.5
    mode: str = "pressure"       # "pressure" | "delta" | "policy"
    decay: float = 0.85          # per-hop attenuation
    cooldown: int = 5            # min dst steps between firings
    last_influence_t: int = -999_999
    hits: int = 0
    applied_sum: float = 0.0
    applied_ema: float = 0.0
    ema_alpha: float = 0.2       # EMA smoothing

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
    """

    def __init__(
        self,
        damping: float = 0.5,
        max_depth: int = 3,
        max_step: float = 1.0,
        rupture_only_propagation: bool = True,
        rng: Optional[np.random.Generator] = None,
        cascade_trace_cap: int = 512,
        # Optional softcap to compress 'base' before scaling (None | "tanh" | "log1p")
        softcap: Optional[str] = None,
        softcap_k: float = 1.0,
    ):
        self.nodes: Dict[str, EpistemicState] = {}
        self.edges: Dict[str, Dict[str, Edge]] = {}
        self.damping = float(damping)
        self.max_depth = int(max_depth)
        self.max_step = float(max_step)
        self.rupture_only = bool(rupture_only_propagation)
        self.graph_rng: np.random.Generator = rng or np.random.default_rng()

        # Rolling trace for debugging / UIs
        self._cascade_trace: List[Dict[str, Any]] = []
        self._cascade_trace_cap = int(max(0, cascade_trace_cap))

        # Global switch to silence propagation
        self._suspended: bool = False

        # Graph-level hooks: event -> list[Callable[[dict], None]]
        self._hooks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

        # Graph event log (lightweight)
        self.event_log: List[Dict[str, Any]] = []
        self._t: int = 0  # coarse graph time

        # Per-node transient policy bias (reversible)
        # name -> {"theta_base": float, "k_base": float, "dtheta": float, "dk": float}
        self._policy_bias: Dict[str, Dict[str, float]] = {}
        self._policy_bias_decay: float = 0.9  # decay each step for that node

        # Optional softcap for base magnitudes
        self.softcap: Optional[str] = softcap
        self.softcap_k: float = float(softcap_k)

    # ---- node & edge management ----

    def add(self, name: str, state: Optional[EpistemicState] = None, **kwargs) -> None:
        """Add a node (construct EpistemicState(**kwargs) if state is None)."""
        if name in self.nodes:
            raise KeyError(f"Node '{name}' already exists")
        if state is None:
            state = EpistemicState(**kwargs)
        self.nodes[name] = state
        self.edges.setdefault(name, {})
        try:
            state._log_event("graph_node_added", {"node": name})
        except Exception:
            pass
        # Reuse on_link channel for node-added telemetry
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
        """Create/overwrite a directed link `src -> dst`."""
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist in graph.")
        if mode not in ("pressure", "delta", "policy"):
            raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
        self.edges.setdefault(src, {})
        self.edges[src][dst] = Edge(
            weight=float(abs(weight)),  # enforce non-negative
            mode=mode,
            decay=float(decay),
            cooldown=int(cooldown),
        )
        try:
            meta = {"src": src, "dst": dst, "mode": mode, "weight": float(abs(weight)), "decay": float(decay), "cooldown": int(cooldown)}
            self.nodes[src]._log_event("graph_edge_linked", meta)
            self.nodes[dst]._log_event("graph_edge_linked", meta)
        except Exception:
            pass
        self._emit("on_link", {"event": "edge_linked", "src": src, "dst": dst, "mode": mode})

    def update_link(self, src: str, dst: str, **updates: Any) -> None:
        """Update parameters of an existing link (weight/decay/cooldown/mode)."""
        e = self.edges.get(src, {}).get(dst)
        if not e:
            raise KeyError(f"No edge {src}->{dst}")
        if "mode" in updates:
            mode = str(updates["mode"])
            if mode not in ("pressure", "delta", "policy"):
                raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
            e.mode = mode
        if "weight" in updates:
            e.weight = float(abs(updates["weight"]))
        if "decay" in updates:
            e.decay = float(updates["decay"])
        if "cooldown" in updates:
            e.cooldown = int(updates["cooldown"])
        try:
            meta = {"src": src, "dst": dst, **{k: updates[k] for k in updates if k in {"weight", "decay", "cooldown", "mode"}}}
            self.nodes[dst]._log_event("graph_edge_updated", meta)
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
        """Return a read-only snapshot of the adjacency map for `src`."""
        return dict(self.edges.get(src, {}))  # note: returns live Edge objects

    # ---- stepping ----

    def step(self, name: str, R: Any) -> Dict[str, Any]:
        """Feed evidence to a node and (optionally) propagate influence."""
        if name not in self.nodes:
            raise KeyError(f"Unknown node '{name}'")
        n = self.nodes[name]

        pre_ruptured = bool(n.last().get("ruptured")) if n.last() else False

        # Decay & apply any pending reversible policy bias before processing
        self._decay_and_apply_policy_bias(name)

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

        # Do not mutate `post` fields; attach diagnostics under a namespaced key
        meta = {"oscillation": ("flip" if ruptured != pre_ruptured else "steady")}
        out = dict(post)
        out["_graph_meta"] = meta

        self._t += 1
        return out

    def step_all(self, evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Batch step: process `{node_name: R}` for each item in `evidence`."""
        return {name: self.step(name, R) for name, R in evidence.items()}

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

            # Edge cooldown (relative to destination node's step counter)
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

            # Optional softcap on base (pre-scale compression)
            if self.softcap == "tanh":
                base = float(np.tanh(self.softcap_k * base))
            elif self.softcap == "log1p":
                base = float(np.log1p(self.softcap_k * base))

            magnitude = float(self.damping * e.weight * base * (e.decay ** (depth - 1)))

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
                applied = self._nudge_policy(dst, magnitude)

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

            # Optional: mirror into node history as a 'nudge' (if kernel supports it)
            if applied != 0.0 and hasattr(dst_state, "_log_nudge"):
                try:
                    dst_state._log_nudge(applied=applied, src=src, mode=e.mode, depth=depth)  # type: ignore[attr-defined]
                except Exception:
                    pass

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
        """Move dst_state.V slightly toward its own last R (preferred) or source R direction."""
        dst_R_vec = self._last_R_scalar_or_vec(dst_state)

        if isinstance(dst_state.V, np.ndarray):
            v = np.asarray(dst_state.V, dtype=float)
            n_v = float(np.linalg.norm(v))

            if dst_R_vec is not None:
                direction = dst_R_vec / (np.linalg.norm(dst_R_vec) or 1.0)
            else:
                if n_v > 0:
                    direction = v / n_v
                else:
                    src_R = src_post.get("R")
                    if isinstance(src_R, (list, np.ndarray)):
                        rvec = np.asarray(src_R, dtype=float)
                        direction = rvec / (np.linalg.norm(rvec) or 1.0)
                    else:
                        direction = np.ones_like(v, dtype=float)
                        direction /= (np.linalg.norm(direction) or 1.0)  # normalize fallback

            step = direction * float(magnitude)

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
                target = float(np.linalg.norm(dst_R_vec))
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

    def _nudge_policy(self, name: str, magnitude: float) -> float:
        """Apply a reversible, decaying bias to Θ (up) and k (down) for node `name`."""
        st = self.nodes[name]
        try:
            # lower bound at 0.0 since magnitude >= 0
            dθ_inc = float(np.clip(0.05 * magnitude, 0.0, 0.2))
            dk_inc = float(np.clip(0.03 * magnitude, 0.0, 0.2))
            bias = self._policy_bias.get(name)
            if bias is None:
                bias = {
                    "theta_base": float(st.Θ),
                    "k_base": float(st.k),
                    "dtheta": 0.0,
                    "dk": 0.0,
                }
                self._policy_bias[name] = bias
            # accumulate with hard caps to avoid runaway
            bias["dtheta"] = float(np.clip(bias["dtheta"] + dθ_inc, -0.5, 0.5))
            bias["dk"]     = float(np.clip(bias["dk"] + dk_inc,  0.0, 0.5))
            # apply immediately
            st.Θ = max(1e-6, bias["theta_base"] + bias["dtheta"])
            st.k = max(1e-3, bias["k_base"] - bias["dk"])
            return abs(dθ_inc) + abs(dk_inc)
        except Exception:
            return 0.0  # keep network resilient

    def _decay_and_apply_policy_bias(self, name: str) -> None:
        """Decay pending policy bias for node and re-apply; restore when negligible."""
        bias = self._policy_bias.get(name)
        if not bias:
            return
        st = self.nodes[name]
        # decay
        bias["dtheta"] *= self._policy_bias_decay
        bias["dk"]     *= self._policy_bias_decay
        # if nearly zero, restore baseline and drop entry
        if abs(bias["dtheta"]) < 1e-6 and abs(bias["dk"]) < 1e-6:
            st.Θ = float(bias["theta_base"])
            st.k = float(bias["k_base"])
            self._policy_bias.pop(name, None)
            return
        # else apply decayed values
        st.Θ = max(1e-6, bias["theta_base"] + bias["dtheta"])
        st.k = max(1e-3, bias["k_base"] - bias["dk"])

    # ---- diagnostics ----

    @staticmethod
    def _oscillation_factor(state: EpistemicState, window: int = 20) -> float:
        """Damping multiplier in [0.5, 1.0] based on rupture flip frequency."""
        hist = state.history[-window:]
        if len(hist) < 4:
            return 1.0
        rupt = np.array([1 if h.get("ruptured", False) else 0 for h in hist], dtype=int)
        flips = np.abs(np.diff(rupt)).sum()
        factor = 1.0 - min(0.5, flips / max(8, window))
        return float(np.clip(factor, 0.5, 1.0))

    def stats(self) -> Dict[str, Any]:
        """Quick aggregate snapshot for dashboards."""
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
        """Return the adjacency with edge metadata for visualization."""
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
            "graph_class": type(self).__name__,
            "damping": self.damping,
            "max_depth": self.max_depth,
            "max_step": self.max_step,
            "rupture_only": self.rupture_only,
            "softcap": self.softcap,
            "softcap_k": self.softcap_k,
            "nodes": list(self.nodes.keys()),
            "edges": {
                src: {
                    dst: e.as_dict() for dst, e in nbrs.items()
                }
                for src, nbrs in self.edges.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_graph(self, path: str) -> None:
        """
        Load adjacency/edge metadata from JSON. Nodes are created if missing
        with default EpistemicState(); node internals are NOT restored.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.damping = float(payload.get("damping", self.damping))
        self.max_depth = int(payload.get("max_depth", self.max_depth))
        self.max_step = float(payload.get("max_step", self.max_step))
        self.rupture_only = bool(payload.get("rupture_only", self.rupture_only))
        self.softcap = payload.get("softcap", self.softcap)
        self.softcap_k = float(payload.get("softcap_k", self.softcap_k))

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
                    weight=float(abs(meta.get("weight", 0.5))),
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
                if e.weight < 0:
                    issues.append(f"negative weight on {src}->{dst}")
                if e.mode not in ("pressure", "delta", "policy"):
                    issues.append(f"invalid mode on {src}->{dst}: {e.mode}")
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
        """Replay a list of evidence dicts (each is {node: R}) with live propagation."""
        out: List[Dict[str, Dict[str, Any]]] = []
        for step_ev in evidence_sequence:
            out.append(self.step_all(step_ev))
        return out

    def predict_influence(self, src: str, dst: str, post: Optional[Dict[str, Any]] = None, rupture: Optional[bool] = None, depth: int = 1) -> float:
        """Compute the capped magnitude that *would* be applied from src->dst; does not mutate state."""
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
        # Apply same softcap path used in _propagate_from
        if self.softcap == "tanh":
            base = float(np.tanh(self.softcap_k * base))
        elif self.softcap == "log1p":
            base = float(np.log1p(self.softcap_k * base))
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

# ======================================================================
# Programmable layer (fully injectable, backward-compatible)
# ======================================================================

VectorLike = Union[List[float], np.ndarray]

# ---- Strategy Protocols (APIs you can implement) ----
class GateFn(Protocol):
    def __call__(self, src_st: EpistemicState, dst_st: EpistemicState, ctx: Dict[str, Any]) -> bool: ...

class InfluenceFn(Protocol):
    def __call__(self, src_st: EpistemicState, post_src: Dict[str, Any], ctx: Dict[str, Any]) -> float: ...

class MagnitudeFn(Protocol):
    def __call__(self, base: float, edge_meta: Dict[str, Any], dst_st: EpistemicState, ctx: Dict[str, Any]) -> float: ...

class TargetFn(Protocol):
    def __call__(self, dst_st: EpistemicState, edge_meta: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[slice]: ...

class NudgeFn(Protocol):
    def __call__(self, dst_st: EpistemicState, magnitude: float, post_src: Dict[str, Any],
                 target: Optional[slice], ctx: Dict[str, Any]) -> float: ...

class DampFn(Protocol):
    def __call__(self, dst_st: EpistemicState, ctx: Dict[str, Any]) -> float: ...

# ---- Default strategies (mirror current behavior) ----
def _default_gate(src_st: EpistemicState, dst_st: EpistemicState, ctx: Dict[str, Any]) -> bool:
    mode = ctx["edge"]["mode"]
    rupt = bool(ctx["post_src"].get("ruptured", False))
    return (mode != "pressure" and mode != "policy") or rupt

def _default_influence(src_st: EpistemicState, post_src: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    mode = ctx["edge"]["mode"]
    delta = float(post_src.get("∆", 0.0))
    theta = float(post_src.get("Θ", 0.0))
    pressure = max(0.0, delta - theta)
    if mode == "pressure": return pressure
    if mode == "delta":    return delta * 0.25
    if mode == "policy":   return pressure
    return 0.0

def _default_magnitude(base: float, edge_meta: Dict[str, Any], dst_st: EpistemicState, ctx: Dict[str, Any]) -> float:
    g = ctx["graph"]
    depth = int(ctx["depth"])
    # softcap (opt-in)
    if getattr(g, "softcap", None) == "tanh":
        base = float(np.tanh(getattr(g, "softcap_k", 1.0) * base))
    elif getattr(g, "softcap", None) == "log1p":
        base = float(np.log1p(getattr(g, "softcap_k", 1.0) * base))
    mag = float(g.damping * edge_meta["weight"] * base * (edge_meta["decay"] ** max(0, depth - 1)))
    # optional spatial/parametric distance decay
    dist_decay = edge_meta.get("dist_decay", 0.0)
    if callable(dist_decay):
        mag *= float(dist_decay(ctx))
    elif dist_decay and dist_decay > 0.0:
        p = getattr(g, "_pos", {})
        p1, p2 = p.get(ctx["src"]), p.get(ctx["dst"])
        if p1 and p2:
            dx, dy = (p1[0]-p2[0]), (p1[1]-p2[1]); dist = float(np.hypot(dx, dy))
            mag *= float(np.exp(-float(dist_decay) * dist))
    node_cap = float(getattr(dst_st, "step_cap", g.max_step))
    cap = float(min(g.max_step, max(1e-6, node_cap)))
    return float(np.clip(mag, -cap, cap))

def _default_target(dst_st: EpistemicState, edge_meta: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[slice]:
    # None => whole V. (i,j) or [i, j] => slice. Or resolver(name) via resolve_channel.
    if not isinstance(dst_st.V, np.ndarray):
        return None
    t = edge_meta.get("target", None)
    if t is None:
        return slice(0, dst_st.V.shape[0])
    if isinstance(t, (tuple, list)) and len(t) == 2:
        i, j = int(t[0]), int(t[1])
        i = max(0, min(i, dst_st.V.shape[0])); j = max(i, min(j, dst_st.V.shape[0]))
        return slice(i, j)
    res = edge_meta.get("resolve_channel")
    if isinstance(t, str) and callable(res):
        return res(dst_st, t, edge_meta, ctx)
    if callable(t):
        return t(dst_st, edge_meta, ctx)
    return None

def _default_damp(dst_st: EpistemicState, ctx: Dict[str, Any]) -> float:
    g = ctx["graph"]
    if hasattr(g, "_oscillation_factor"):
        return float(g._oscillation_factor(dst_st))
    return 1.0

def _default_nudge(dst_st: EpistemicState, magnitude: float, post_src: Dict[str, Any],
                   target: Optional[slice], ctx: Dict[str, Any]) -> float:
    g = ctx["graph"]
    if isinstance(dst_st.V, np.ndarray) and target is not None:
        v = np.asarray(dst_st.V, dtype=float); sub = v[target]
        n_sub = float(np.linalg.norm(sub))
        last = dst_st.last() or {}
        R = last.get("R", None)
        if isinstance(R, (list, np.ndarray)) and len(R) == len(v):
            r_sub = np.asarray(R, dtype=float)[target]
            direction = r_sub / (np.linalg.norm(r_sub) or 1.0)
        else:
            direction = (sub / (n_sub or 1.0)) if n_sub > 0 else np.ones_like(sub)
        step = direction * float(magnitude)
        node_cap = float(getattr(dst_st, "step_cap", g.max_step))
        cap = float(min(g.max_step, max(1e-6, node_cap)))
        step_norm = float(np.linalg.norm(step))
        if step_norm > cap:
            step *= (cap / (step_norm or 1.0)); step_norm = cap
        v[target] = (sub + step).astype(float); dst_st.V = v
        return step_norm
    # fallback to scalar/whole-vector logic
    return g._nudge_value_toward_recent_R(dst_st, magnitude, post_src)

# ---- Programmable Edge ----
@dataclass
class ProgrammableEdge(Edge):
    gate_fn: Optional[GateFn] = None
    influence_fn: Optional[InfluenceFn] = None
    magnitude_fn: Optional[MagnitudeFn] = None
    target_fn: Optional[TargetFn] = None
    nudge_fn: Optional[NudgeFn] = None
    damp_fn: Optional[DampFn] = None
    params: Dict[str, Any] = field(default_factory=dict)
    # passthrough fields used by defaults
    target: Any = None
    resolve_channel: Optional[Callable[..., Optional[slice]]] = None
    dist_decay: Any = 0.0
    # persistence by reference
    strategy_id: Optional[str] = None

# ---- Fully-injectable graph ----
class EpistemicProgrammableGraph(EpistemicGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defaults: Dict[str, Callable] = {
            "gate_fn":      _default_gate,
            "influence_fn": _default_influence,
            "magnitude_fn": _default_magnitude,
            "target_fn":    _default_target,
            "nudge_fn":     _default_nudge,
            "damp_fn":      _default_damp,
        }
        self._pos: Dict[str, Tuple[float, float]] = {}
        self.edges: Dict[str, Dict[str, ProgrammableEdge]] = {}

    # Override to accept callables without breaking the original API
    def link(self, src: str, dst: str, weight: float = 0.5, mode: str = "pressure",
             decay: float = 0.85, cooldown: int = 5, **plug) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist in graph.")
        if mode not in ("pressure", "delta", "policy"):
            raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
        self.edges.setdefault(src, {})
        edge = ProgrammableEdge(
            weight=float(abs(weight)), mode=mode, decay=float(decay), cooldown=int(cooldown),
            gate_fn=plug.get("gate_fn"), influence_fn=plug.get("influence_fn"),
            magnitude_fn=plug.get("magnitude_fn"), target_fn=plug.get("target_fn"),
            nudge_fn=plug.get("nudge_fn"), damp_fn=plug.get("damp_fn"),
            params=dict(plug.get("params", {})),
            target=plug.get("target", None),
            resolve_channel=plug.get("resolve_channel", None),
            dist_decay=plug.get("dist_decay", 0.0),
            strategy_id=plug.get("strategy_id"),
        )
        self.edges[src][dst] = edge
        try:
            meta = {"src": src, "dst": dst, "mode": mode, "weight": float(abs(weight)),
                    "decay": float(decay), "cooldown": int(cooldown), "plug": list(plug.keys()),
                    "strategy_id": edge.strategy_id}
            self.nodes[src]._log_event("graph_edge_linked", meta)
            self.nodes[dst]._log_event("graph_edge_linked", meta)
        except Exception:
            pass
        self._emit("on_link", {"event": "edge_linked", "src": src, "dst": dst, "mode": mode})

    def neighbors(self, src: str) -> Dict[str, ProgrammableEdge]:
        return dict(self.edges.get(src, {}))

    # Hot-update programmable callables/params while preserving base updates
    def update_link(self, src: str, dst: str, **updates) -> None:
        e = self.edges.get(src, {}).get(dst)
        if not e:
            raise KeyError(f"No edge {src}->{dst}")
        # First apply base scalar fields via parent
        base_updates = {k: v for k, v in updates.items() if k in {"mode", "weight", "decay", "cooldown"}}
        if base_updates:
            super().update_link(src, dst, **base_updates)
        # Then programmable fields
        for key in ("gate_fn", "influence_fn", "magnitude_fn", "target_fn", "nudge_fn", "damp_fn",
                    "target", "resolve_channel", "dist_decay", "strategy_id"):
            if key in updates:
                setattr(e, key, updates[key])
        if "params" in updates and isinstance(updates["params"], dict):
            e.params.update(updates["params"])
        self._emit("on_link", {"event": "edge_updated", "src": src, "dst": dst,
                               "plug_updates": [k for k in updates.keys()
                                                if k not in {"mode", "weight", "decay", "cooldown"}]})

    def set_default(self, name: str, fn: Callable) -> None:
        if name not in self.defaults:
            raise KeyError(f"Unknown default '{name}'")
        self.defaults[name] = fn

    def set_position(self, name: str, x: float, y: float) -> None:
        self._pos[name] = (float(x), float(y))

    # Core propagation using injected strategies
    def _propagate_from(self, src: str, depth: int, rupture: bool) -> None:
        if depth > self.max_depth:
            return
        post_src = self.nodes[src].last() or {}
        for dst, e in self.neighbors(src).items():
            dst_st = self.nodes[dst]
            if (dst_st.summary()["t"] - e.last_influence_t) < e.cooldown:
                continue

            ctx = {
                "graph": self, "src": src, "dst": dst, "depth": depth, "time": self._t,
                "post_src": post_src,
                "edge": {
                    "mode": e.mode, "weight": e.weight, "decay": e.decay, "cooldown": e.cooldown,
                    "target": e.target, "resolve_channel": e.resolve_channel, "dist_decay": e.dist_decay,
                    "params": e.params,
                },
                "rng": self.graph_rng,
            }

            gate_fn = e.gate_fn or self.defaults["gate_fn"]
            if not gate_fn(self.nodes[src], dst_st, ctx):
                continue

            influence_fn = e.influence_fn or self.defaults["influence_fn"]
            base = float(influence_fn(self.nodes[src], post_src, ctx))
            if base <= 0.0:
                continue

            damp_fn = e.damp_fn or self.defaults["damp_fn"]
            damp = float(np.clip(damp_fn(dst_st, ctx), 0.0, 1.0))

            magnitude_fn = e.magnitude_fn or self.defaults["magnitude_fn"]
            mag = float(magnitude_fn(base, ctx["edge"], dst_st, ctx)) * damp
            if mag == 0.0:
                continue

            target_fn = e.target_fn or self.defaults["target_fn"]
            t_slice = target_fn(dst_st, ctx["edge"], ctx)

            nudge_fn = e.nudge_fn or self.defaults["nudge_fn"]
            applied = float(nudge_fn(dst_st, mag, post_src, t_slice, ctx))

            if applied != 0.0:
                e.hits += 1
                e.applied_sum += abs(applied)
                e.applied_ema = float(e.ema_alpha * abs(applied) + (1.0 - e.ema_alpha) * e.applied_ema)

            event = {
                "t": self._t, "src": src, "dst": dst, "depth": depth, "mode": e.mode,
                "base": base, "magnitude": mag, "applied": applied, "target": e.target,
                "params": e.params, "strategy_id": e.strategy_id,
            }
            self._record_cascade(event)
            try:
                dst_st._log_event("graph_influence", event)
            except Exception:
                pass
            # Optional: mirror into node history as a 'nudge' if available
            if applied != 0.0 and hasattr(dst_st, "_log_nudge"):
                try:
                    dst_st._log_nudge(applied=applied, src=src, mode=e.mode, depth=depth)  # type: ignore[attr-defined]
                except Exception:
                    pass
            self._emit("on_influence", event)

            e.last_influence_t = dst_st.summary()["t"]
            # Recurse to next hop
            self._propagate_from(dst, depth + 1, rupture=bool(post_src.get("ruptured", False)))

    # ---- Persistence overrides (include strategies by reference) ----

    def save_graph(self, path: str, include_strategies: bool = True) -> None:
        """
        Persist adjacency/edge metadata to JSON. For programmable edges, also saves:
          - strategy_id (string, if set)
          - params (JSON-safe dict; callables removed, numpy types converted)
          - target (JSON-safe; callables dropped)
          - dist_decay (numeric; callables dropped -> 0.0)
        Callables are NOT serialized.
        """
        payload = {
            "graph_class": type(self).__name__,
            "damping": self.damping,
            "max_depth": self.max_depth,
            "max_step": self.max_step,
            "rupture_only": self.rupture_only,
            "softcap": self.softcap,
            "softcap_k": self.softcap_k,
            "nodes": list(self.nodes.keys()),
            "edges": {},
        }
        for src, nbrs in self.edges.items():
            payload["edges"][src] = {}
            for dst, e in nbrs.items():
                base = e.as_dict()
                if include_strategies and isinstance(e, ProgrammableEdge):
                    # Guard target / dist_decay against callables; deep-sanitize params
                    raw_params = e.params if isinstance(e.params, dict) else {}
                    safe_params = _json_sanitize(raw_params)
                    if not isinstance(safe_params, dict):
                        safe_params = {}
                    tgt = e.target if not callable(e.target) else None
                    dd = e.dist_decay if not callable(e.dist_decay) else 0.0
                    dd = _json_sanitize(dd)
                    base.update({
                        "strategy_id": e.strategy_id,
                        "params": safe_params,
                        "target": _json_sanitize(tgt),
                        "dist_decay": dd if _is_json_scalar(dd) else 0.0,
                    })
                payload["edges"][src][dst] = base
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_graph(self, path: str, *, strict_strategies: bool = False) -> None:
        """
        Load adjacency/edge metadata from JSON. Rebinds programmable strategies by ID.
        If a strategy is missing:
          - strict_strategies=True -> raises KeyError
          - else -> falls back to defaults and logs 'graph_strategy_missing'
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.damping = float(payload.get("damping", self.damping))
        self.max_depth = int(payload.get("max_depth", self.max_depth))
        self.max_step = float(payload.get("max_step", self.max_step))
        self.rupture_only = bool(payload.get("rupture_only", self.rupture_only))
        self.softcap = payload.get("softcap", self.softcap)
        self.softcap_k = float(payload.get("softcap_k", self.softcap_k))

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
                e = ProgrammableEdge(
                    weight=float(abs(meta.get("weight", 0.5))),
                    mode=str(meta.get("mode", "pressure")),
                    decay=float(meta.get("decay", 0.85)),
                    cooldown=int(meta.get("cooldown", 5)),
                    params=dict(meta.get("params", {})),
                    target=meta.get("target", None),
                    dist_decay=meta.get("dist_decay", 0.0),
                    strategy_id=meta.get("strategy_id"),
                )
                # counters
                e.last_influence_t = int(meta.get("last_influence_t", -999_999))
                e.hits = int(meta.get("hits", 0))
                e.applied_sum = float(meta.get("applied_sum", 0.0))
                e.applied_ema = float(meta.get("applied_ema", 0.0))
                e.ema_alpha = float(meta.get("ema_alpha", 0.2))

                # Rebind callables from registry if strategy_id present
                sid = e.strategy_id
                if sid:
                    fns = get_strategy(sid)
                    if not fns and strict_strategies:
                        raise KeyError(f"Missing strategy '{sid}' in registry.")
                    if fns:
                        for k, fn in fns.items():
                            setattr(e, k, fn)
                    else:
                        # Loud but non-fatal
                        try:
                            self.nodes[dst]._log_event("graph_strategy_missing", {"edge": f"{src}->{dst}", "strategy_id": sid})
                        except Exception:
                            pass

                self.edges[src][dst] = e
