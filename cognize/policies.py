# Prebuilt rupture, collapse, and realign functions for injection into EpistemicState

import numpy as np

### ------------------------
### Collapse Strategies
### ------------------------

def collapse_reset(R, V, E):
    """Default hard reset to zero."""
    return 0.0, 0.0

def collapse_soft_decay(R, V, E, gamma=0.5, beta=0.3):
    """Soft decay of projection and memory."""
    return V * gamma, E * beta

def collapse_adopt_R(R, V, E):
    """Collapse to adopt external reality as new projection."""
    return R, 0.0

def collapse_randomized(R, V, E):
    """Collapse to a random small perturbation."""
    return np.random.normal(0, 0.1), 0.0

# Self-compatible lambda wrappers
collapse_reset_fn         = lambda self: collapse_reset(None, self.V, self.E)
collapse_soft_decay_fn    = lambda self: collapse_soft_decay(None, self.V, self.E)
collapse_adopt_R_fn       = lambda self: collapse_adopt_R(self.V, self.V, self.E)
collapse_randomized_fn    = lambda self: collapse_randomized(self.V, self.V, self.E)

### ------------------------
### Realignment Kernels (⊙ variants)
### ------------------------

def realign_linear(V, delta, E, k):
    """Default RCC linear realignment."""
    return V + k * delta * (1 + E)

def realign_tanh(V, delta, E, k):
    """Tanh-bounded realignment (slows as delta grows)."""
    return V + np.tanh(k * delta) * (1 + E)

def realign_bounded(V, delta, E, k, cap=1.0):
    """Caps max projection shift per step."""
    shift = min(k * delta * (1 + E), cap)
    return V + shift

def realign_decay_adaptive(V, delta, E, k):
    """Realign with adaptive gain decaying as E increases."""
    gain = k / (1 + E)
    return V + gain * delta

# Self-compatible lambda wrappers
realign_linear_fn          = lambda self, R, d: realign_linear(self.V, d, self.E, self.k)
realign_tanh_fn            = lambda self, R, d: realign_tanh(self.V, d, self.E, self.k)
realign_bounded_fn         = lambda self, R, d: realign_bounded(self.V, d, self.E, self.k)
realign_decay_adaptive_fn  = lambda self, R, d: realign_decay_adaptive(self.V, d, self.E, self.k)

### ------------------------
### Threshold Functions (Θ(t))
### ------------------------

def threshold_static(E, t, base=0.35):
    """Fixed threshold regardless of time or memory."""
    return base

def threshold_adaptive(E, t, base=0.35, a=0.05):
    """Linear adaptation to misalignment memory."""
    return base + a * E

def threshold_stochastic(E, t, base=0.35, sigma=0.02):
    """Adds stochastic noise to baseline threshold."""
    return base + np.random.normal(0, sigma)

def threshold_combined(E, t, base=0.35, a=0.05, sigma=0.01):
    """Adaptive + stochastic threshold model."""
    return base + a * E + np.random.normal(0, sigma)

# Self-compatible lambda wrappers
threshold_static_fn     = lambda self: threshold_static(self.E, self._time)
threshold_adaptive_fn   = lambda self: threshold_adaptive(self.E, self._time)
threshold_stochastic_fn = lambda self: threshold_stochastic(self.E, self._time)
threshold_combined_fn   = lambda self: threshold_combined(self.E, self._time)

