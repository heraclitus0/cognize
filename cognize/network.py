# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.network
===============

EpistemicGraph — orchestrates multiple EpistemicState nodes as a directed,
influence-aware graph.

What it gives you
-----------------
- **Multi-node coordination**: manage many `EpistemicState` objects together.
- **Rupture-aware propagation**: when a node ruptures (∆ > Θ), push influence to neighbors.
- **Modes of influence**:
  - `"pressure"`: propagate rupture pressure (∆ − Θ) to neighbors as a nudge.
  - `"delta"`: continuous, weaker coupling even without rupture (based on ∆).
  - `"policy"`: under stress, gently tighten neighbor thresholds and reduce gain `k`.
- **Safety**:
  - Per-edge cooldown to avoid thrashing.
  - Depth-limited cascades with per-hop decay.
  - Oscillation guard dampens influence on nodes that keep flipping rupture state.
  - Bounded steps respect both the graph cap and each node’s `step_cap`.

Quick start
-----------
>>> from cognize.epistemic import EpistemicState
>>> from cognize.network import EpistemicGraph
>>> g = EpistemicGraph(damping=0.5, max_depth=2)
>>> g.add("price", state=EpistemicState(threshold=0.3))
>>> g.add("risk", state=EpistemicState(threshold=0.4))
>>> g.link("price", "risk", weight=0.6, mode="pressure")
>>> g.step("price", 1.2)  # feed evidence to 'price'
>>> g.stats()             # quick snapshot for dashboards

Design notes
------------
- This module does **not** mutate policies permanently; `"policy"` mode just nudges live
  runtime parameters (Θ and k) slightly under stress. Your own policy system can still
  govern long-lived changes.
- Works with scalar or vector `V`. For vectors, nudges are directional and bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    last_influence_t: int = -999_999


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
        `"delta"` edges still couple (weakly) without rupture. Default True.
    """

    def __init__(
        self,
        damping: float = 0.5,
        max_depth: int = 3,
        max_step: float = 1.0,
        rupture_only_propagation: bool = True,
    ):
        self.nodes: Dict[str, EpistemicState] = {}
        self.edges: Dict[str, Dict[str, Edge]] = {}
        self.damping = float(damping)
        self.max_depth = int(max_depth)
        self.max_step = float(max_step)
        self.rupture_only = bool(rupture_only_propagation)

    # ---- node & edge management ----

    def add(self, name: str, state: Optional[EpistemicState] = None, **kwargs) -> None:
        """
        Add a node.

        Parameters
        ----------
        name : str
            Node name (must be unique).
        state : EpistemicState, optional
            An existing state to insert. If None, a new one is constructed from kwargs.
        **kwargs :
            Passed to `EpistemicState(**kwargs)` when `state` is None.
        """
        if state is None:
            state = EpistemicState(**kwargs)
        self.nodes[name] = state
        self.edges.setdefault(name, {})

    def link(
        self,
        src: str,
        dst: str,
        weight: float = 0.5,
        mode: str = "pressure",
        decay: float = 0.85,
        cooldown: int = 5,
    ) -> None:
        """
        Create/overwrite a directed link `src -> dst`.

        Parameters
        ----------
        src, dst : str
            Node names (must both exist).
        weight : float
            Influence multiplier. Typical range [0.0, 1.0].
        mode : {"pressure", "delta", "policy"}
            Influence semantics:
            - "pressure": propagate (∆ − Θ) when src ruptures.
            - "delta": continuous weak coupling proportional to ∆ (no rupture needed).
            - "policy": on rupture, nudge dst's Θ and reduce k slightly (reversible).
        decay : float
            Per-hop attenuation in cascades. Typical 0.7–0.95.
        cooldown : int
            Minimum steps between consecutive influences along this edge.
        """
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist in graph.")
        if mode not in ("pressure", "delta", "policy"):
            raise ValueError("mode must be one of: 'pressure', 'delta', 'policy'")
        self.edges[src][dst] = Edge(
            weight=float(weight),
            mode=mode,
            decay=float(decay),
            cooldown=int(cooldown),
        )

    def unlink(self, src: str, dst: str) -> None:
        """Remove the directed link `src -> dst` if it exists."""
        self.edges.get(src, {}).pop(dst, None)

    def neighbors(self, src: str) -> Dict[str, Edge]:
        """Return the adjacency map for `src`."""
        return self.edges.get(src, {})

    # ---- stepping ----

    def step(self, name: str, R: Any) -> Dict[str, Any]:
        """
        Feed evidence to a node and (optionally) propagate influence.

        Parameters
        ----------
        name : str
            Node name to step.
        R : Any
            Evidence, passed directly to `EpistemicState.receive`.

        Returns
        -------
        dict
            The node's last log entry (convenience for dashboards).
        """
        if name not in self.nodes:
            raise KeyError(f"Unknown node '{name}'")
        n = self.nodes[name]

        # Pre-step rupture status (for oscillation note)
        pre_ruptured = bool(n.last().get("ruptured")) if n.last() else False

        n.receive(R, source=name)
        post = n.last() or {}
        ruptured = bool(post.get("ruptured", False))

        if self.rupture_only and not ruptured:
            # Allow continuous coupling only for "delta" edges
            self._propagate_from(name, depth=1, rupture=False)
        else:
            self._propagate_from(name, depth=1, rupture=ruptured)

        post["_osc_note"] = "flip" if ruptured != pre_ruptured else "steady"
        return post

    def step_all(self, evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Batch step: process `{node_name: R}` for each item in `evidence`.

        Nodes without evidence are skipped.

        Returns
        -------
        dict
            `{node_name: last_log_entry}`
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, R in evidence.items():
            out[name] = self.step(name, R)
        return out

    def broadcast(self, R: Any, nodes: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Feed the same evidence to many nodes (e.g., common shocks).

        Parameters
        ----------
        R : Any
            Evidence forwarded to each node.
        nodes : Iterable[str], optional
            Subset to broadcast to. Defaults to all nodes.

        Returns
        -------
        dict
            `{node_name: last_log_entry}`
        """
        targets = list(nodes) if nodes else list(self.nodes.keys())
        return {name: self.step(name, R) for name in targets}

    # ---- propagation core ----

    def _propagate_from(self, src: str, depth: int, rupture: bool) -> None:
        """
        Recursively propagate influence from `src` up to `max_depth`.
        """
        if depth > self.max_depth:
            return
        post = self.nodes[src].last() or {}
        delta = float(post.get("∆", 0.0))
        theta = float(post.get("Θ", 0.0))
        pressure = max(0.0, delta - theta)  # rupture pressure; zero if no rupture

        for dst, e in self.neighbors(src).items():
            dst_state = self.nodes[dst]

            # Edge cooldown check
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
            if e.mode in ("pressure", "delta"):
                self._nudge_value_toward_recent_R(dst_state, magnitude, src_post=post)
            elif e.mode == "policy" and magnitude > 0.0:
                self._nudge_policy(dst_state, magnitude)

            # Mark last influence step for cooldown
            e.last_influence_t = dst_state.summary()["t"]

            # Recurse
            self._propagate_from(dst, depth + 1, rupture=rupture)

    # ---- influence primitives ----

    @staticmethod
    def _last_R_scalar_or_vec(state: EpistemicState) -> Optional[np.ndarray]:
        """
        Fetch last seen R as a vector if possible; fallback to 1-D vector from scalar.
        """
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
    ) -> None:
        """
        Move `dst_state.V` a little toward its own last `R` (preferred) or toward
        the source node’s `R` direction as a fallback. Works for scalars and vectors.
        The step is bounded by both graph and node caps.
        """
        dst_R_vec = self._last_R_scalar_or_vec(dst_state)

        if isinstance(dst_state.V, np.ndarray):
            v = np.asarray(dst_state.V, dtype=float)
            if dst_R_vec is None:
                # Fallback to source R direction if available
                src_R = src_post.get("R")
                if isinstance(src_R, (list, np.ndarray)):
                    rvec = np.asarray(src_R, dtype=float)
                    direction = rvec / (np.linalg.norm(rvec) or 1.0)
                else:
                    direction = np.ones_like(v)
            else:
                direction = dst_R_vec / (np.linalg.norm(dst_R_vec) or 1.0)

            step = direction * float(magnitude)

            # Respect destination node’s cap as well
            node_cap = float(getattr(dst_state, "step_cap", self.max_step))
            cap = float(min(self.max_step, max(1e-6, node_cap)))
            step_norm = float(np.linalg.norm(step))
            if step_norm > cap:
                step = step * (cap / (step_norm or 1.0))

            dst_state.V = (v + step).astype(float)

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
            dst_state.V = float(dst_state.V) + sign * float(np.clip(magnitude, -cap, cap))

    @staticmethod
    def _nudge_policy(dst_state: EpistemicState, magnitude: float) -> None:
        """
        Light-touch runtime bias: slightly increase Θ and reduce k under stress.
        Reversible; meant for transient stabilization, not persistent mutation.
        """
        try:
            dst_state.Θ = max(
                1e-6, float(dst_state.Θ + np.clip(0.05 * magnitude, -0.2, 0.2))
            )
            dst_state.k = float(
                max(1e-3, dst_state.k - np.clip(0.03 * magnitude, 0.0, 0.2))
            )
        except Exception:
            # Keep graph resilient to odd node states
            pass

    # ---- diagnostics ----

    @staticmethod
    def _oscillation_factor(state: EpistemicState, window: int = 20) -> float:
        """
        Compute a damping multiplier in [0.5, 1.0] based on rupture flip frequency.
        More flips -> smaller factor.
        """
        hist = state.history[-window:]
        if len(hist) < 4:
            return 1.0
        rupt = np.array([1 if h.get("ruptured", False) else 0 for h in hist], dtype=int)
        flips = np.abs(np.diff(rupt)).sum()
        factor = 1.0 - min(0.5, flips / max(8, window))
        return float(np.clip(factor, 0.5, 1.0))

    def stats(self) -> Dict[str, Any]:
        """
        Quick aggregate snapshot for dashboards.

        Returns
        -------
        dict
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

    def adjacency(self) -> Dict[str, Dict[str, dict]]:
        """
        Return the adjacency with edge metadata for visualization.

        Returns
        -------
        dict
            {src: {dst: {"weight", "mode", "decay", "cooldown"}}}
        """
        adj: Dict[str, Dict[str, dict]] = {}
        for src, nbrs in self.edges.items():
            adj[src] = {
                dst: {
                    "weight": e.weight,
                    "mode": e.mode,
                    "decay": e.decay,
                    "cooldown": e.cooldown,
                }
                for dst, e in nbrs.items()
            }
        return adj

