# Stub file for Cognize public API

from typing import Any, Callable, Optional
import numpy as np

# Core
class EpistemicState:
    def __init__(
        self,
        V0: float = 0.0,
        threshold: float = 0.35,
        realign_strength: float = 0.3,
        rng_seed: Optional[int] = None
    ) -> None: ...
    def receive(self, R: float, source: Optional[str] = None) -> None: ...
    def last(self) -> dict[str, Any]: ...

class EpistemicGraph:
    def __init__(self, damping: float = 0.5) -> None: ...
    def add(self, name: str, **kwargs: Any) -> None: ...
    def link(self, src: str, dst: str) -> None: ...
    def step(self, name: str, R: float) -> None: ...

class Perception:
    def __init__(
        self,
        text_encoder: Optional[Callable[[str], np.ndarray]] = None,
        image_encoder: Optional[Callable[[Any], np.ndarray]] = None,
        sensor_fusion_fn: Optional[Callable[[dict[str, Any]], np.ndarray]] = None
    ) -> None: ...
    def process(self, inputs: dict[str, Any]) -> np.ndarray: ...

# Meta-learning
class PolicySpec: ...
class PolicyMemory: ...
class ShadowRunner: ...
class PolicyManager:
    def __init__(
        self,
        safe_specs: list[PolicySpec],
        memory: PolicyMemory,
        runner: ShadowRunner,
        epsilon: float = 0.15,
        promote_margin: float = 1.03,
        cooldown_steps: int = 30
    ) -> None: ...

SAFE_SPECS: list[PolicySpec]

# Policies
POLICY_REGISTRY: dict[str, Callable[..., Any]]

# Defaults
def realign_tanh_fn(self, R: float, d: float) -> float: ...
def threshold_adaptive_fn(self) -> float: ...
def collapse_soft_decay_fn(self) -> tuple[float, float]: ...

# Convenience
def make_simple_state(
    V0: float = 0.0,
    threshold: float = 0.35,
    realign_strength: float = 0.3,
    seed: Optional[int] = None,
    with_meta: bool = False
) -> EpistemicState: ...

__version__: str
__author__: str
__license__: str
