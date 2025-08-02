"""
Cognize: A symbolic cognition engine for programmable epistemic drift tracking.

Tracks epistemic projection (V), incoming reality (R), misalignment memory (E),
rupture (Θ, ∆), and supports programmable realignment (⊙), collapse (⇓), and symbolic rupture (⚠).

Built for agents, simulations, and drift-aware systems.

Public API exposes:
- EpistemicState: core cognition engine
- policies: full symbolic injection module
- 3 symbolic defaults: ⊙ (realign), Θ (threshold), ⇓ (collapse)
"""

from .epistemic import EpistemicState           # Core cognitive engine
from . import policies                          # Namespaced access for all injection functions

# Symbolic default policies (minimal curated surface)
from .policies import (
    realign_tanh_fn,            # ⊙: bounded realignment
    threshold_adaptive_fn,      # Θ(t): dynamic threshold adjustment
    collapse_soft_decay_fn      # ⇓: partial memory-preserving rupture
)

# Public API contract (intentional exposure)
__all__ = [
    "EpistemicState",           # core engine
    "policies",                 # full symbolic injection module
    "realign_tanh_fn",          # ⊙
    "threshold_adaptive_fn",    # Θ
    "collapse_soft_decay_fn"    # ⇓
]
