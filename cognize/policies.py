# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-
"""
cognize.policies
================
Prebuilt, safe-by-default policies for EpistemicState.

Signatures (aligned with upgraded EpistemicState):
- Threshold:  fn(state) -> float
- Realign:    fn(state, R_val: float, delta: float) -> float          # scalar-V path
- Collapse:   fn(state, R_val: float | None) -> (V_new: float, E_new: float)

Notes
-----
- Vector states are handled inside EpistemicState; realign/collapse here target the
  scalar path. For vectors, EpistemicState performs bounded directional steps.
- All functions apply conservative bounds to prevent runaway behavior.
"""

from __future__ import annotations

from typing import Dict, Callable, Tuple, Optional, Any
import numpy as np

# ---------------------------
# RNG helper (deterministic with EpistemicState.rng)
# ---------------------------

def _rng(state):
    """Return the state's local RNG (preferred) or a fresh default_rng()."""
    return getattr(state, "rng", np.random.default_rng())

# --------------
# Threshold (Θ)
# --------------

def threshold_static(state) -> float:
    """Fixed threshold; uses state's current Θ as base."""
    return float(state.Θ)

def threshold_adaptive(state, a: float = 0.05, cap: float = 1.0) -> float:
    """Θ_t = base + a * E (bounded)."""
    a = float(np.clip(a, 0.0, 1.0))
    cap = float(np.clip(cap, 0.0, 10.0))
    return float(state.Θ + np.clip(a * float(state.E), 0.0, cap))

def threshold_stochastic(state, sigma: float = 0.01) -> float:
    """Baseline Θ with small Gaussian exploration noise (deterministic via state.rng)."""
    sigma = float(np.clip(sigma, 0.0, 0.2))
    return float(state.Θ + float(_rng(state).normal(0.0, sigma)))

def threshold_combined(state, a: float = 0.05, sigma: float = 0.01, cap: float = 1.0) -> float:
    """Adaptive + stochastic; both parts bounded."""
    a = float(np.clip(a, 0.0, 1.0))
    sigma = float(np.clip(sigma, 0.0, 0.2))
    adapt = float(np.clip(a * float(state.E), 0.0, cap))
    return float(state.Θ + adapt + float(_rng(state).normal(0.0, sigma)))

# --------------
# Realign (⊙)
# --------------

def realign_linear(state, R_val: float, delta: float) -> float:
    """
    Linear step toward R with bounded gain.
    V' = V + sign(R - V) * clip(k * delta * (1 + E), ±step_cap)
    """
    k = float(np.clip(getattr(state, "k", 0.3), 0.0, 2.0))
    step = k * float(delta) * (1.0 + float(state.E))
    cap = float(getattr(state, "step_cap", 1.0))
    step = float(np.clip(step, -cap, cap))
    sign = 1.0 if (float(R_val) - float(state.V)) >= 0.0 else -1.0
    return float(state.V) + sign * step

def realign_tanh(state, R_val: float, delta: float) -> float:
    """Tanh-bounded step (slower for large deltas) and E-bounded via tanh(E/eps)."""
    k = float(np.clip(getattr(state, "k", 0.3), 0.0, 2.0))
    eps = float(max(getattr(state, "epsilon", 1e-3), 1e-9))
    gain = float(np.tanh(k * float(delta))) * (1.0 + float(np.tanh(float(state.E) / eps)))
    cap = float(getattr(state, "step_cap", 1.0))
    gain = float(np.clip(gain, -cap, cap))
    sign = 1.0 if (float(R_val) - float(state.V)) >= 0.0 else -1.0
    return float(state.V) + sign * gain

def realign_bounded(state, R_val: float, delta: float, cap: float = 1.0) -> float:
    """Cap absolute shift per step independent of state.step_cap (uses min of the two)."""
    k = float(np.clip(getattr(state, "k", 0.3), 0.0, 2.0))
    step = k * float(delta) * (1.0 + float(state.E))
    cap = float(max(1e-6, min(float(cap), float(getattr(state, "step_cap", 1.0)))))
    step = float(np.clip(step, -cap, cap))
    sign = 1.0 if (float(R_val) - float(state.V)) >= 0.0 else -1.0
    return float(state.V) + sign * step

def realign_decay_adaptive(state, R_val: float, delta: float) -> float:
    """Gain decays with E: k' = k/(1+E), then bounded by step_cap."""
    k = float(np.clip(getattr(state, "k", 0.3) / (1.0 + float(state.E)), 0.0, 1.0))
    step = k * float(delta)
    cap = float(getattr(state, "step_cap", 1.0))
    step = float(np.clip(step, -cap, cap))
    sign = 1.0 if (float(R_val) - float(state.V)) >= 0.0 else -1.0
    return float(state.V) + sign * step

# --------------
# Collapse (post-Θ)
# --------------

def collapse_reset(state, R_val: Optional[float] = None) -> Tuple[float, float]:
    """Hard reset to zero (scalar path); E→0."""
    return 0.0, 0.0

def collapse_soft_decay(state, R_val: Optional[float] = None,
                        gamma: float = 0.5, beta: float = 0.3) -> Tuple[float, float]:
    """Soften both V and E without adopting R directly: V' = γV, E' = βE."""
    gamma = float(np.clip(gamma, 0.0, 1.0))
    beta  = float(np.clip(beta, 0.0, 1.0))
    return float(state.V) * gamma, float(state.E) * beta

def collapse_adopt_R(state, R_val: Optional[float] = None) -> Tuple[float, float]:
    """Adopt scalar R as new projection; E→0."""
    rv = float(R_val if R_val is not None else state.V)
    return rv, 0.0

def collapse_randomized(state, R_val: Optional[float] = None, sigma: float = 0.1) -> Tuple[float, float]:
    """Jump to a small random neighborhood; E→0. Uses state.rng for determinism."""
    sigma = float(np.clip(sigma, 0.0, 1.0))
    return float(_rng(state).normal(0.0, sigma)), 0.0

# --------------
# Legacy-compatible helpers (old call shapes, if anything imports them)
# --------------

def _collapse_reset_legacy(R, V, E):                                     return 0.0, 0.0
def _collapse_soft_decay_legacy(R, V, E, gamma=0.5, beta=0.3):            return V * gamma, E * beta
def _collapse_adopt_R_legacy(R, V, E):                                    return R, 0.0
def _collapse_randomized_legacy(R, V, E, sigma=0.1):                      return float(np.random.normal(0.0, sigma)), 0.0

def _realign_linear_legacy(V, delta, E, k):           return V + k * delta * (1 + E)
def _realign_tanh_legacy(V, delta, E, k):             return V + np.tanh(k * delta) * (1 + E)
def _realign_bounded_legacy(V, delta, E, k, cap=1.):  return V + min(k * delta * (1 + E), cap)
def _realign_decay_adaptive_legacy(V, delta, E, k):   return V + (k / (1 + E)) * delta

def _threshold_static_legacy(E, t, base=0.35):                 return float(base)
def _threshold_adaptive_legacy(E, t, base=0.35, a=0.05):       return float(base + a * E)
def _threshold_stochastic_legacy(E, t, base=0.35, sigma=0.02): return float(base + float(np.random.normal(0, sigma)))
def _threshold_combined_legacy(E, t, base=0.35, a=0.05, sigma=0.01):
    return float(base + a * E + float(np.random.normal(0, sigma)))

# --------------
# Kernel-compatible wrappers (back-compat with EpistemicState.inject_policy)
# --------------

threshold_static_fn     = lambda state: threshold_static(state)
threshold_adaptive_fn   = lambda state: threshold_adaptive(state)
threshold_stochastic_fn = lambda state: threshold_stochastic(state)
threshold_combined_fn   = lambda state: threshold_combined(state)

realign_linear_fn          = lambda state, R, d: realign_linear(state, float(R), float(d))
realign_tanh_fn            = lambda state, R, d: realign_tanh(state, float(R), float(d))
realign_bounded_fn         = lambda state, R, d: realign_bounded(state, float(R), float(d))
realign_decay_adaptive_fn  = lambda state, R, d: realign_decay_adaptive(state, float(R), float(d))

collapse_reset_fn       = lambda state, R=None: collapse_reset(state, R)
collapse_soft_decay_fn  = lambda state, R=None: collapse_soft_decay(state, R)
collapse_adopt_R_fn     = lambda state, R=None: collapse_adopt_R(state, R)
collapse_randomized_fn  = lambda state, R=None: collapse_randomized(state, R)

# --------------
# Registry (optional convenience)
# --------------

REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "threshold": {
        "static": threshold_static,
        "adaptive": threshold_adaptive,
        "stochastic": threshold_stochastic,
        "combined": threshold_combined,
    },
    "realign": {
        "linear": realign_linear,
        "tanh": realign_tanh,
        "bounded": realign_bounded,
        "decay_adaptive": realign_decay_adaptive,
    },
    "collapse": {
        "reset": collapse_reset,
        "soft_decay": collapse_soft_decay,
        "adopt_R": collapse_adopt_R,
        "randomized": collapse_randomized,
    },
}

__all__ = [
    # current API
    "threshold_static", "threshold_adaptive", "threshold_stochastic", "threshold_combined",
    "realign_linear", "realign_tanh", "realign_bounded", "realign_decay_adaptive",
    "collapse_reset", "collapse_soft_decay", "collapse_adopt_R", "collapse_randomized",
    "REGISTRY",
    # kernel-compatible wrappers (back-compat)
    "threshold_static_fn", "threshold_adaptive_fn", "threshold_stochastic_fn", "threshold_combined_fn",
    "realign_linear_fn", "realign_tanh_fn", "realign_bounded_fn", "realign_decay_adaptive_fn",
    "collapse_reset_fn", "collapse_soft_decay_fn", "collapse_adopt_R_fn", "collapse_randomized_fn",
]
