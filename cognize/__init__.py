# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-

"""
Cognize
=======
Belief dynamics middleware: EpistemicState + Policies + Meta-learning + Graph.

Quick start:
    from cognize import EpistemicState, POLICY_REGISTRY, SAFE_SPECS
    s = EpistemicState(V0=0.0)

Meta-learning one-liner:
    from cognize import make_simple_state
    s = make_simple_state(with_meta=True)
"""

# Version/metadata
try:
    from .epistemic import __version__ as __version__
except Exception:
    __version__ = "0.2.0-pre"
__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache 2.0"

# Core kernel & satellites
from .epistemic import (
    EpistemicState,
    Perception,
    PolicyManager,
    PolicySpec,
    PolicyMemory,
    ShadowRunner,
    SAFE_SPECS,
)

from .network import EpistemicGraph

# Policies
from .policies import REGISTRY as POLICY_REGISTRY

# Namespaces for power users
from . import policies as policies
from . import network as network
from . import meta_learning as meta_learning

# Convenience factory
def make_simple_state(
    V0=0.0,
    threshold: float = 0.35,
    realign_strength: float = 0.3,
    seed: int | None = None,
    with_meta: bool = False,
):
    state = EpistemicState(V0=V0, threshold=threshold, realign_strength=realign_strength, rng_seed=seed)
    if with_meta:
        pm = PolicyManager(SAFE_SPECS, PolicyMemory(), ShadowRunner(), epsilon=0.15, promote_margin=1.03, cooldown_steps=30)
        state.policy_manager = pm
    return state

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
