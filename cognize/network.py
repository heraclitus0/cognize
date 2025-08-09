# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.network
===============

EpistemicGraph — orchestrates multiple EpistemicState nodes in a directed graph.
- Weighted links with damping & decay
- Rupture-pressure propagation (with depth & cooldown controls)
- Vector-safe nudging and optional policy-trigger propagation
- Batch stepping and broadcast convenience helpers
- Light safety guards (oscillation & storm prevention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional, List, Any, Iterable
import numpy as np

from .epistemic import EpistemicState


# ---------------------------
# Edge definition
# ---------------------------

@dataclass
class Edge:
    """Directed influence from src -> dst."""
    weight: float = 0.5          # base multiplier on rupture pressure
    mode: str = "pressure"       # "pressure" | "delta" | "policy"
    decay: float = 0.85          # per-hop attenuation when cascading
    cooldown: int = 5            # min steps between repeated strong influences
    last_influence_t: int = -999999


# ---------------------------
# Graph
# ---------------------------

class EpistemicGraph:
    """
    A small orchestrator for multiple EpistemicState nodes with influence links.

    Influence semantics
    -------------------
    - mode="pressure" : push dst.V toward its *last* R direction by a step
      proportional to (src.∆ - src.Θ)*weight (clamped) when src ruptures.
    - mode="delta"    : push dst based on sign of (src.R - src.V) regardless of rupture,
      scaled down strongly (continuous coupling).
    - mode="policy"   : on src rupture, nudge dst's threshold/realign strength
      (light-touch; reversible drift dampener).

    Safety
    ------
    - Depth-limited cascades with per-edge decay.
    - Per-edge cooldown to avoid thrash.
    - Oscillation guard: if dst flips rupture state too often recently, reduce influence.
    """

    def __init__(self,
                 damping: float = 0.5,
                 max_depth: int = 3,
                 max_step: float = 1.0,
                 rupture_only_propagation: bool = True):
        self.nodes: Dict[str, EpistemicState] = {}
        self.edges: Dict[str, Dict[str, Edge]] = {}
        self.damping = float(damping)
        self.max_depth = int(max_depth)
        self.max_step = float(max_step)
        self.rupture_only = bool(rupture_only_propagation)

    # ---- node & edge management ----

    def add(self, name: str, state: Optional[EpistemicState] = None, **kwargs) -> None:
        """
        Add a node. Pass an EpistemicState or kwargs to construct one.
        """
        if state is None:
            state = EpistemicState(**kwargs)
        self.nodes[name] = state
        self.edges.setdefault(name, {})

    def link(self, src: str, dst: str, weight: float = 0.5, mode: str = "pressure",
             decay: float = 0.85, cooldown: int = 5) -> None:
        """
        Create/overwrite a directed link src -> dst.
        """
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist in graph.")
        if mode not in ("pressure", "delta", "policy"):
            raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
        self.edges[src][dst] = Edge(weight=float(weight), mode=mode, decay=float(decay), cooldown=int(cooldown))

    def unlink(self, src: str, dst: str) -> None:
        self.edges.get(src, {}).pop(dst, None)

    def neighbors(self, src: str) -> Dict[str, Edge]:
        return self.edges.get(src, {})

    # ---- stepping ----

    def step(self, name: str, R: Any) -> Dict[str, Any]:
        """
        Feed evidence to a node and possibly propagate influence.
        Returns the node's last log entry for convenience.
        """
        if name not in self.nodes:
            raise KeyError(f"Unknown node '{name}'")
        n = self.nodes[name]

        # Pre-step rupture status (for oscillation analysis)
        pre_ruptured = bool(n.last().get("ruptured")) if n.last() else False

        n.receive(R, source=name)
        post = n.last() or {}
        ruptured = bool(post.get("ruptured", False))

        if self.rupture_only and not ruptured:
            # Opportunistic continuous coupling could still run for "delta" edges
            self._propagate_from(name, depth=1, rupture=False)
        else:
            self._propagate_from(name, depth=1, rupture=ruptured)

        # Optional oscillation note
        post["_osc_note"] = f"{'flip' if ruptured != pre_ruptured else 'steady'}"
        return post

    def step_all(self, evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Batch step: evidence is {node_name: R}. Returns {node_name: last_log}.
        Nodes without evidence are skipped (no op).
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, R in evidence.items():
            out[name] = self.step(name, R)
        return out

    def broadcast(self, R: Any, nodes: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Feed the same evidence to many nodes, useful for common shocks or shared inputs.
        """
        targets = list(nodes) if nodes else list(self.nodes.keys())
        return {name: self.step(name, R) for name in targets}

    # ---- propagation core ----

    def _propagate_from(self, src: str, depth: int, rupture: bool) -> None:
        """
        Internal: recursively propagate influence from src up to max_depth.
        """
        if depth > self.max_depth:
            return
        post = self.nodes[src].last() or {}
        delta = float(post.get("∆", 0.0))
        theta = float(post.get("Θ", 0.0))
        pressure = max(0.0, delta - theta)  # rupture pressure; zero if no rupture

        for dst, e in self.neighbors(src).items():
            dst_state = self.nodes[dst]
            # Cooldown
            if (dst_state.summary()["t"] - e.last_influence_t) < e.cooldown:
                continue

            # Influence magnitude (bounded)
            if e.mode == "pressure":
                if not rupture:
                    continue
                base = pressure
            elif e.mode == "delta":
                # continuous coupling uses delta even without rupture (soft)
                base = delta * 0.25
            else:  # "policy"
                base = pressure if rupture else 0.0

            if base <= 0.0:
                continue

            magnitude = float(np.clip(self.damping * e.weight * base * (e.decay ** (depth - 1)),
                                      -self.max_step, self.max_step))

            # Oscillation guard on dst: if dst is flipping too often, reduce influence
            osc_factor = self._oscillation_factor(dst_state)
            magnitude *= osc_factor

            # Apply according to mode
            if e.mode in ("pressure", "delta"):
                self._nudge_value_toward_recent_R(dst_state, magnitude, src_post=post)
            elif e.mode == "policy" and magnitude > 0.0:
                self._nudge_policy(dst_state, magnitude)

            e.last_influence_t = dst_state.summary()["t"]

            # Recurse
            self._propagate_from(dst, depth + 1, rupture=rupture)

    # ---- influence primitives ----

    @staticmethod
    def _last_R_scalar_or_vec(state: EpistemicState) -> Optional[np.ndarray]:
        """Fetch last seen R as a vector if possible; fallback to scalar wrapped."""
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

    def _nudge_value_toward_recent_R(self, dst_state: EpistemicState, magnitude: float, src_post: Dict[str, Any]) -> None:
        """
        Move dst.V a little toward its own last R (if available) or toward src's R direction as a fallback.
        Works for scalars and vectors, keeps steps bounded.
        """
        # Prefer dst's own last R
        dst_R_vec = self._last_R_scalar_or_vec(dst_state)

        if isinstance(dst_state.V, np.ndarray):
            v = np.asarray(dst_state.V, dtype=float)
            if dst_R_vec is None:
                # Fallback to source R direction magnitude if vector available
                src_R = src_post.get("R")
                if isinstance(src_R, (list, np.ndarray)):
                    rvec = np.asarray(src_R, dtype=float)
                    direction = rvec / (np.linalg.norm(rvec) or 1.0)
                else:
                    direction = np.ones_like(v)
            else:
                direction = dst_R_vec / (np.linalg.norm(dst_R_vec) or 1.0)

            step = direction * float(magnitude)
            # Cap by graph max_step additionally
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.max_step:
                step = step * (self.max_step / (step_norm or 1.0))

            dst_state.V = (v + step).astype(float)

        else:
            # Scalar case
            if dst_R_vec is not None:
                target = float(dst_R_vec.mean())
            else:
                src_R = src_post.get("R")
                try:
                    target = float(src_R if not isinstance(src_R, (list, np.ndarray)) else np.linalg.norm(src_R))
                except Exception:
                    target = float(dst_state.V)

            sign = 1.0 if (target - float(dst_state.V)) >= 0.0 else -1.0
            dst_state.V = float(dst_state.V) + sign * float(np.clip(magnitude, -self.max_step, self.max_step))

    @staticmethod
    def _nudge_policy(dst_state: EpistemicState, magnitude: float) -> None:
        """
        Light-touch policy bias: slightly tighten threshold or reduce k under stress.
        Reversible: only applied via runtime (no permanent param mutation).
        """
        # Tighten Θ a bit and reduce k, within safe mini-bounds
        try:
            dst_state.Θ = float(dst_state.Θ + np.clip(0.05 * magnitude, -0.2, 0.2))
            dst_state.k = float(max(0.01, dst_state.k - np.clip(0.03 * magnitude, 0.0, 0.2)))
        except Exception:
            pass  # keep graph resilient

    # ---- diagnostics ----

    @staticmethod
    def _oscillation_factor(state: EpistemicState, window: int = 20) -> float:
        """
        Compute a damping multiplier in [0.5, 1.0] based on rupture flip frequency.
        """
        hist = state.history[-window:]
        if len(hist) < 4:
            return 1.0
        rupt = np.array([1 if h.get("ruptured", False) else 0 for h in hist], dtype=int)
        flips = np.abs(np.diff(rupt)).sum()
        # More flips -> smaller factor
        factor = 1.0 - min(0.5, flips / max(8, window))
        return float(np.clip(factor, 0.5, 1.0))

    def stats(self) -> Dict[str, Any]:
        """
        Quick aggregate view for dashboards.
        """
        out = {}
        for name, s in self.nodes.items():
            ds = s.drift_stats(window=min(50, len(s.history)))
            out[name] = {
                "ruptures": s.summary()["ruptures"],
                "mean_drift": ds.get("mean_drift", 0.0),
                "std_drift": ds.get("std_drift", 0.0),
                "last_symbol": s.symbol(),
            }
        return out

    def adjacency(self) -> Dict[str, Dict[str, dict]]:
        """
        Return the adjacency with edge metadata for visualization.
        """
        adj: Dict[str, Dict[str, dict]] = {}
        for src, nbrs in self.edges.items():
            adj[src] = {dst: {"weight": e.weight, "mode": e.mode, "decay": e.decay, "cooldown": e.cooldown}
                        for dst, e in nbrs.items()}
        return adj
