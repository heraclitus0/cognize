# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-

"""
GenesisAgent — rupture-aware, self-evolving controller
================================================================

What this example demonstrates (Cognize UVP):
  • Epistemic kernel: EpistemicState tracks belief V vs reality R with misalignment memory E and Θ-based ruptures.
  • Runtime policy programming: swap threshold/realign/collapse policies on the fly.
  • Safe meta-policy evolution: PolicyManager + ShadowRunner evaluate candidates in shadow, promote only if they win.
  • Multimodal Perception: custom fuse() turns {text/image/sensor} dicts into a normalized vector.
  • Auditability: CSV traces and event summaries.

Run (from repo root):
    pip install -e .
    python -m cognize.examples.genesis_agent --steps 260 --prefix demo

NOTE: Examples are NOT part of the public API surface; treat as patterns to build your own agents.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List

import numpy as np

from cognize import (
    __version__,
    EpistemicState, Perception,
    PolicyManager, PolicyMemory, ShadowRunner,
    SAFE_SPECS, POLICY_REGISTRY,
)


# ----------------------------
# Toy multimodal encoders
# ----------------------------
def text_encoder(s: str) -> np.ndarray:
    s = str(s)
    vowels = sum(s.lower().count(v) for v in "aeiou")
    digits = sum(ch.isdigit() for ch in s)
    punct = sum(ch in ".,;:!?-\"'" for ch in s)
    return np.array([len(s), s.count(" "), vowels, digits, punct, 1.0], dtype=float)

def image_encoder(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=float)
    return np.array([
        float(x.mean()),
        float(x.std() + 1e-9),
        float(x.max()) if x.size else 0.0,
        float(x.min()) if x.size else 0.0,
        float((x**2).mean()) if x.size else 0.0,
        1.0
    ], dtype=float)

def sensor_encoder(d: Dict[str, Any]) -> np.ndarray:
    t = float(d.get("temp", 0.0))
    v = float(d.get("vib", 0.0))
    f = float(d.get("flow", 0.0))
    return np.array([t, v, f, t*f, v*f, 1.0], dtype=float)


# ----------------------------
# Custom fusion for Perception
# ----------------------------
def fuse_all(modalities: Dict[str, Any]) -> np.ndarray:
    """
    Concatenate whichever modalities are present; Perception can normalize downstream.
    """
    parts: List[np.ndarray] = []
    if "text" in modalities and modalities["text"] is not None:
        parts.append(text_encoder(modalities["text"]))
    if "image" in modalities and modalities["image"] is not None:
        parts.append(image_encoder(modalities["image"]))
    if "sensor" in modalities and modalities["sensor"] is not None:
        parts.append(sensor_encoder(modalities["sensor"]))
    if not parts:
        parts.append(np.array([0, 0, 0, 0, 0, 1.0], dtype=float))  # bias-only fallback
    return np.concatenate(parts, axis=0)


# ----------------------------
# Helpers for safe policy lookup
# ----------------------------
def _get_policy(name: str):
    """
    Support both styles:
      - Flat registry: POLICY_REGISTRY['collapse_soft_decay']
      - Namespaced registry: POLICY_REGISTRY['collapse']['collapse_soft_decay']
    """
    if name in POLICY_REGISTRY:
        return POLICY_REGISTRY[name]
    if "collapse" in POLICY_REGISTRY and name in POLICY_REGISTRY["collapse"]:
        return POLICY_REGISTRY["collapse"][name]
    return None


def tighten_after_rupture(state: EpistemicState,
                          theta_floor: float = 0.18,
                          k_cap: float = 0.45,
                          k_boost: float = 1.15) -> None:
    """
    On rupture: toggle collapse behavior and tighten Θ / boost k within safe bounds.
    """
    ev = state.last()
    if not ev or not ev.get("rupture", False):
        return

    # Toggle collapse between two canonical behaviors if available
    soft = _get_policy("collapse_soft_decay")
    adopt = _get_policy("collapse_adopt_R")
    current = state.policies.get("collapse") if hasattr(state, "policies") else None
    next_collapse = adopt if current is soft else soft or adopt
    if next_collapse is not None:
        state.inject_policy(collapse=next_collapse)

    # Tighten Θ and boost k conservatively
    try:
        state.threshold = max(theta_floor, float(state.threshold) * 0.90)
    except Exception:
        pass
    try:
        state.realign_strength = min(k_cap, float(state.realign_strength) * k_boost)
    except Exception:
        pass


# ----------------------------
# Wrap-ready agent
# ----------------------------
class GenesisAgent:
    """
    Two EpistemicStates:
      - fast_state: reacts quickly to fused evidence
      - slow_state: more conservative defaults for stability

    Public API:
      - step(evidence: dict) -> tuple[dict, dict]: returns last event dicts
      - summarize() -> str
      - export(prefix: str) -> None
    """

    def __init__(self,
                 threshold_fast: float = 0.35,
                 threshold_slow: float = 0.38,
                 k_fast: float = 0.25,
                 k_slow: float = 0.20,
                 seed_fast: int = 7,
                 seed_slow: int = 11,
                 normalize: bool = True,
                 explain: bool = True):
        self.perception = Perception(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            fuse=fuse_all,
            normalize=normalize,
            explain=explain
        )

        # Fast / slow epistemic controllers
        self.fast_state = EpistemicState(
            V0=np.zeros(6, dtype=float),  # initial shape; auto-handled by Perception
            threshold=threshold_fast,
            realign_strength=k_fast,
            perception=self.perception,
            rng_seed=seed_fast,
            name="Fast"
        )
        self.slow_state = EpistemicState(
            V0=np.zeros(6, dtype=float),
            threshold=threshold_slow,
            realign_strength=k_slow,
            perception=self.perception,
            rng_seed=seed_slow,
            name="Slow"
        )

        # Safe meta-policy evolution on both states
        for st in (self.fast_state, self.slow_state):
            st.policy_manager = PolicyManager(
                base_specs=SAFE_SPECS,
                memory=PolicyMemory(),
                shadow=ShadowRunner(),
                epsilon=0.12,            # bounded exploration
                promote_margin=1.025,    # ≥ +2.5% to promote
                cooldown_steps=25        # avoid thrash
            )
            st.enable_auto_evolution(
                param_space={
                    "conservative": {"k": (0.12, 0.28), "Θ": (0.24, 0.55)},
                    "cautious":     {"k": (0.10, 0.25), "Θ": (0.22, 0.60)},
                    "adoptive":     {"k": (0.20, 0.38), "Θ": (0.20, 0.62)},
                },
                every=30, rate=1.0, margin=1.02
            )

        self._promotions = 0
        self._ruptures = 0

    def step(self, evidence: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Accepts dicts like:
            {"text": "...", "image": np.ndarray, "sensor": {"temp":..., "vib":..., "flow":...}}
        Returns last event dicts from (fast, slow).
        """
        self.fast_state.receive(evidence)
        self.slow_state.receive(evidence)

        # Rupture-aware tightening
        tighten_after_rupture(self.fast_state)
        tighten_after_rupture(self.slow_state)

        # Track events
        for st in (self.fast_state, self.slow_state):
            ev = st.last()
            if ev.get("rupture", False):
                self._ruptures += 1
            if ev.get("event") == "policy_promoted":
                self._pro_
