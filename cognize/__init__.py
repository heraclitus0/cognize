# Licensed under the Apache License, Version 2.0
# -*- coding: utf-8 -*-

"""
Cognize
=======
Belief dynamics middleware: EpistemicState + Policies + Meta-learning + Graph.

Ergonomic entrypoint:

    import cognize as cg

Common imports:
    from cognize import EpistemicState, EpistemicGraph, Perception
    from cognize import PolicyManager, PolicySpec, PolicyMemory, ShadowRunner
    from cognize import POLICY_REGISTRY, SAFE_SPECS
    from cognize import EpistemicProgrammableGraph, register_strategy, get_strategy
    from cognize import PerceptionConfig, PerceptionError, RunningCalibrator
    from cognize import ParamRange, ParamSpace, enable_evolution, enable_dynamic_evolution
    from cognize import (threshold_static, threshold_adaptive, threshold_stochastic, threshold_combined,
                         realign_linear, realign_tanh, realign_bounded, realign_decay_adaptive,
                         collapse_reset, collapse_soft_decay, collapse_adopt_R, collapse_randomized)

Versioning:
    __version__ is single-sourced from `cognize.epistemic` (>= 0.1.8). If that import fails,
    the local fallback is used. Releases MUST keep both in lock-step.

Import behavior:
    - Top-level names above are eagerly imported for ergonomics and IDE discovery.
    - Subpackages `policies`, `network`, and `meta_learning` are lazy-loaded on first
      attribute access via `__getattr__` (e.g., `cognize.policies`).
"""

from __future__ import annotations

# --- Package metadata ----------------------------------------------------------
__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache-2.0"

try:  # prefer kernel version (single source of truth)
    from .epistemic import __version__ as __version__
except Exception:  # pragma: no cover
    __version__ = "0.1.8"

# --- Core kernel & satellites --------------------------------------------------
from .epistemic import (
    EpistemicState,
    Perception,           # optional perception adapter (re-exported by kernel)
    PolicyManager,        # runtime meta-policy selector
    PolicySpec,
    PolicyMemory,
    ShadowRunner,
    SAFE_SPECS,           # preset safe PolicySpec list
)

# Perception configs & errors
from .perception import (
    PerceptionConfig,
    PerceptionError,
    RunningCalibrator,
)

# Base & Programmable graphs (+ registry helpers)
from .network import (
    EpistemicGraph,
    EpistemicProgrammableGraph,
    register_strategy,
    get_strategy,
)

# Policies (prebuilt functions) + registry
from .policies import (
    REGISTRY as POLICY_REGISTRY,
    threshold_static, threshold_adaptive, threshold_stochastic, threshold_combined,
    realign_linear, realign_tanh, realign_bounded, realign_decay_adaptive,
    collapse_reset, collapse_soft_decay, collapse_adopt_R, collapse_randomized,
)

# Meta-learning façade (bounds DSL & enablers)
from .meta_learning import (
    ParamRange,
    ParamSpace,
    enable_evolution,
    enable_dynamic_evolution,
)

# --- Ergonomic helpers ---------------------------------------------------------
import numpy as _np
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def demo_text_encoder(s: str) -> _np.ndarray:
    """
    Tiny, deterministic encoder for demos/tests (no external deps).
    Features: [len, spaces, vowels, digits, has_punct, bias]
    Returns L2-normalized vector.
    """
    s = str(s or "")
    spaces = s.count(" ")
    vowels = sum(ch.lower() in "aeiou" for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    punct = int(any(ch in ".,;:!?-—_()[]{}\"'" for ch in s))
    v = _np.array([len(s), spaces, vowels, digits, punct, 1.0], dtype=float)
    n = float(_np.linalg.norm(v)) or 1.0
    return (v / n).astype(float)


def make_simple_state(
    V0: Union[float, Sequence[float]] = 0.0,
    threshold: float = 0.35,
    realign_strength: float = 0.3,
    seed: Optional[int] = None,
    with_meta: bool = False,
) -> EpistemicState:
    """
    Quick-start factory:
      - Creates EpistemicState with sane defaults
      - Optionally wires a PolicyManager preloaded with SAFE_SPECS
    """
    state = EpistemicState(V0=V0, threshold=threshold, realign_strength=realign_strength, rng_seed=seed)
    if with_meta:
        pm = PolicyManager(SAFE_SPECS, PolicyMemory(), ShadowRunner(),
                           epsilon=0.15, promote_margin=1.03, cooldown_steps=30)
        state.policy_manager = pm
    return state


def make_graph(
    names: Sequence[str],
    edges: Optional[Sequence[Union[
        Tuple[str, str],
        Tuple[str, str, float],
        Tuple[str, str, float, str],
        Tuple[str, str, float, str, float],
        Tuple[str, str, float, str, float, int],
        Tuple[str, str, float, str, float, int, dict],
    ]]] = None,
    *,
    programmable: bool = False,
    with_meta: bool = False,
    seed: Optional[int] = None,
    **graph_kwargs: Any,
) -> Union[EpistemicGraph, EpistemicProgrammableGraph]:
    """
    Construct a (programmable) graph quickly.

    edges: sequence of tuples:
        (src, dst)
        (src, dst, weight)
        (src, dst, weight, mode)
        (src, dst, weight, mode, decay)
        (src, dst, weight, mode, decay, cooldown)
        (src, dst, weight, mode, decay, cooldown, extra_kwargs_dict)  # e.g., strategy_id for programmable
    """
    rng = _np.random.default_rng(seed)
    G = (EpistemicProgrammableGraph if programmable else EpistemicGraph)(
        rng=rng, **graph_kwargs
    )
    for n in names:
        G.add(str(n))

    if edges:
        for e in edges:
            if len(e) == 2:
                src, dst = e
                G.link(src, dst)
            elif len(e) == 3:
                src, dst, w = e
                G.link(src, dst, weight=float(w))
            elif len(e) == 4:
                src, dst, w, mode = e
                G.link(src, dst, weight=float(w), mode=str(mode))
            elif len(e) == 5:
                src, dst, w, mode, decay = e
                G.link(src, dst, weight=float(w), mode=str(mode), decay=float(decay))
            elif len(e) == 6:
                src, dst, w, mode, decay, cooldown = e
                G.link(src, dst, weight=float(w), mode=str(mode), decay=float(decay), cooldown=int(cooldown))
            elif len(e) == 7:
                src, dst, w, mode, decay, cooldown, kw = e
                kw = dict(kw or {})
                if programmable:
                    G.link(src, dst, weight=float(w), mode=str(mode),
                           decay=float(decay), cooldown=int(cooldown), **kw)
                else:
                    # Ignore programmable extras on the plain graph to avoid TypeError
                    G.link(src, dst, weight=float(w), mode=str(mode),
                           decay=float(decay), cooldown=int(cooldown))
            else:
                raise ValueError(f"Unsupported edge spec: {e}")

    if with_meta:
        # Attach a lightweight PolicyManager to each node for auto-policy improvements if desired
        for st in G.nodes.values():
            if st.policy_manager is None:
                st.policy_manager = PolicyManager(SAFE_SPECS, PolicyMemory(), ShadowRunner(),
                                                  epsilon=0.15, promote_margin=1.03, cooldown_steps=30)
    return G


# --- Small UX niceties (monkey-patched, safe & optional) ----------------------

def _state_repr(self: EpistemicState) -> str:  # pragma: no cover (presentation sugar)
    try:
        s = self.summary()
        v = s.get("V")
        v_str = (f"{v:.4f}" if isinstance(v, float) else f"vec(d={len(v)})")
        return (f"EpistemicState(id={s.get('id')}, t={s.get('t')}, V={v_str}, "
                f"E={s.get('E'):.4f}, Θ={s.get('Θ'):.3f}, ruptures={s.get('ruptures')}, "
                f"symbol={s.get('last_symbol')})")
    except Exception:
        return object.__repr__(self)


def _state_observe(self: EpistemicState, R: Any) -> Dict[str, Any]:
    self.receive(R)
    return self.last() or {}


def _state_observe_many(self: EpistemicState, seq: Iterable[Any]) -> Dict[str, Any]:
    for x in seq:
        self.receive(x)
    return self.last() or {}


if not hasattr(EpistemicState, "observe"):
    EpistemicState.observe = _state_observe  # type: ignore[attr-defined]
if not hasattr(EpistemicState, "observe_many"):
    EpistemicState.observe_many = _state_observe_many  # type: ignore[attr-defined]
if EpistemicState.__repr__ is object.__repr__:
    EpistemicState.__repr__ = _state_repr  # type: ignore[assignment]


def _graph_pulse(self: EpistemicGraph, k: int = 5) -> Dict[str, Any]:
    stats = self.stats()
    hot = self.top_edges(by="applied_ema", k=k)
    return {
        "nodes": len(self.nodes),
        "edges": sum(len(v) for v in self.edges.values()),
        "ruptures": {n: stats[n]["ruptures"] for n in stats},
        "mean_drift": {n: round(stats[n]["mean_drift"], 6) for n in stats},
        "hot_edges": [{"src": s, "dst": d, "score": float(v)} for (s, d, v) in hot],
    }


if not hasattr(EpistemicGraph, "pulse"):
    EpistemicGraph.pulse = _graph_pulse  # type: ignore[attr-defined]

# --- Lazy-load deep submodules on first attribute access ----------------------
# Keeps import snappy while retaining discoverability.
import importlib as _importlib
_lazy_submods = {"policies": ".policies", "network": ".network", "meta_learning": ".meta_learning"}


def __getattr__(name: str):
    if name in _lazy_submods:
        mod = _importlib.import_module(_lazy_submods[name], __name__)
        globals()[name] = mod  # cache for subsequent access
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_submods.keys()))

# --- Public API surface for IDEs / star-imports (discouraged but complete) ----
__all__ = [
    # Core
    "EpistemicState",
    "EpistemicGraph",
    "EpistemicProgrammableGraph",
    "Perception",
    # Meta-learning
    "PolicyManager",
    "PolicySpec",
    "PolicyMemory",
    "ShadowRunner",
    "SAFE_SPECS",
    # Perception config/errors
    "PerceptionConfig",
    "PerceptionError",
    "RunningCalibrator",
    # Policies
    "POLICY_REGISTRY",
    "threshold_static", "threshold_adaptive", "threshold_stochastic", "threshold_combined",
    "realign_linear", "realign_tanh", "realign_bounded", "realign_decay_adaptive",
    "collapse_reset", "collapse_soft_decay", "collapse_adopt_R", "collapse_randomized",
    # Graph strategy registry helpers
    "register_strategy", "get_strategy",
    # Bounds DSL + helpers
    "ParamRange", "ParamSpace", "enable_evolution", "enable_dynamic_evolution",
    # Convenience
    "make_simple_state", "make_graph", "demo_text_encoder",
    # Meta
    "__version__", "__author__", "__license__",
]
