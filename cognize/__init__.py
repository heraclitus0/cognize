# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-

"""
Cognize
=======
Belief dynamics middleware: EpistemicState + Policies + Meta-learning + Graph.

Top-level imports (most common):
    from cognize import EpistemicState, EpistemicGraph, Perception
    from cognize import PolicyManager, PolicySpec, PolicyMemory, ShadowRunner
    from cognize import POLICY_REGISTRY, SAFE_SPECS

Namespaced (advanced):
    from cognize import policies, network, meta_learning
"""

from __future__ import annotations

# Version
__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache 2.0"

# Prefer to source version from the kernel; fallback here if needed.
try:
    from .epistemic import __version__ as __version__
except Exception:  # pragma: no cover
    __version__ = "0.2.0-pre"

# --- Core kernel & satellites (top-level ergonomic exports) ---

from .epistemic import (
    EpistemicState,
    Perception,           # optional perception adapter
    PolicyManager,        # runtime meta-policy selector
    PolicySpec,
    PolicyMemory,
    ShadowRunner,
    SAFE_SPECS,           # preset safe PolicySpec list
)

from .network import EpistemicGraph

# --- Policies (prebuilt functions) ---

from .policies import REGISTRY as POLICY_REGISTRY  # {"threshold": {...}, "realign": {...}, "collapse": {...}}

# Also expose the module under a namespace for power users:
from . import policies as policies
from . import network as network
from . import meta_learning as meta_learning

# --- Minimal convenience factory ------------------------------------------------

def make_simple_state(
    V0=0.0,
    threshold: float = 0.35,
    realign_strength: float = 0.3,
    seed: int | None = None,
    with_meta: bool = False,
):
    """
    Quick-start factory:
      - Creates EpistemicState with sane defaults
      - Optionally wires a PolicyManager preloaded with SAFE_SPECS
    """
    state = EpistemicState(V0=V0, threshold=threshold, realign_strength=realign_strength, rng_seed=seed)
    if with_meta:
        pm = PolicyManager(SAFE_SPECS, PolicyMemory(), ShadowRunner(), epsilon=0.15, promote_margin=1.03, cooldown_steps=30)
        state.policy_manager = pm
    return state

# --- Public API surface for IDEs / star-imports ---------------------------------

__all__ = [
    # Core
    "EpistemicState",
    "EpistemicGraph",
    "Perception",
    # Meta-learning
    "PolicyManager",
    "PolicySpec",
    "PolicyMemory",
    "ShadowRunner",
    "SAFE_SPECS",
    # Policies
    "POLICY_REGISTRY",
    # Convenience
    "make_simple_state",
    # Namespaces
    "policies",
    "network",
    "meta_learning",
    # Meta
    "__version__",
    "__author__",
    "__license__",
]
