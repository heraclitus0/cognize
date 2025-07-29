import numpy as np
import uuid
from datetime import datetime

class EpistemicState:
    """
    Cognize — Symbolic cognition tracker and control layer for any agent or system.

    Tracks projection vs. reality, detects epistemic rupture, retains misalignment memory,
    and allows dynamic symbolic intervention.

    Supports: 
    - rupture risk monitoring
    - memory decay
    - binding to external systems (LLMs, sensors, etc.)
    - programmable symbolic intervention
    - projection identity
    - multi-reality (R₁, R₂, …) convergence
    - meta rupture tracking
    """

    def __init__(self, 
                 V0=0.0, 
                 E0=0.0, 
                 threshold=0.35, 
                 realign_strength=0.3,
                 decay_rate=0.9,
                 identity=None,
                 log_history=True):
        # Epistemic fields
        self.V = V0
        self.E = E0
        self.Θ = threshold
        self.k = realign_strength
        self.decay_rate = decay_rate
        self._context_fn = None

        # Internal
        self.history = []
        self.meta_ruptures = []
        self._time = 0
        self._id = str(uuid.uuid4())[:8]
        self.identity = identity or {}
        self._rupture_count = 0
        self._last_symbol = "∅"
        self._triggers = {}

        # Log setting
        self._log = log_history

    def receive(self, R, source="default"):
        """Receive one or multiple reality signals R (float or list)"""
        R_val = self._resolve_reality(R)
        delta = abs(R_val - self.V)
        ruptured = delta > self.Θ
        self._last_symbol = "⚠" if ruptured else "⊙"

        if ruptured:
            self.V, self.E = 0.0, 0.0
            self._rupture_count += 1
            self.meta_ruptures.append({
                "time": self._time,
                "rupture_pressure": delta - self.Θ,
                "source": source
            })
            self._trigger("on_rupture")
        else:
            self.V += self.k * delta * (1 + self.E)
            self.E += 0.1 * delta
            self.E *= self.decay_rate

        if self._log:
            self.history.append({
                "t": self._time,
                "V": self.V,
                "R": R_val,
                "delta": delta,
                "Θ": self.Θ,
                "ruptured": ruptured,
                "symbol": self._last_symbol,
                "source": source
            })

        self._time += 1

    def _resolve_reality(self, R):
        if isinstance(R, (int, float)):
            return R
        elif isinstance(R, list):
            return sum(R) / len(R)
        else:
            raise ValueError("Reality must be float or list of floats")

    def rupture_risk(self):
        if not self.history:
            return None
        return self.history[-1]['delta'] - self.Θ

    def should_intervene(self, margin=0.0):
        risk = self.rupture_risk()
        return risk is not None and risk > margin

    def intervene_if_ruptured(self, fallback_fn, margin=0.0):
        if self.should_intervene(margin):
            return fallback_fn()
        return None

    def reset(self):
        self.V, self.E = 0.0, 0.0
        self.history.clear()
        self._rupture_count = 0
        self._time = 0
        self._last_symbol = "∅"
        self.meta_ruptures.clear()

    def realign(self, R):
        R_val = self._resolve_reality(R)
        self.V = R_val
        self.E *= 0.5
        self._last_symbol = "⊙"
        self.history.append({
            "t": self._time,
            "V": self.V,
            "R": R_val,
            "delta": 0.0,
            "Θ": self.Θ,
            "ruptured": False,
            "symbol": "⊙",
            "source": "manual_realign"
        })
        self._time += 1

    def bind_context(self, fn):
        self._context_fn = fn

    def run_context(self, *args, **kwargs):
        if not self._context_fn:
            raise ValueError("No context function bound.")
        return self._context_fn(*args, **kwargs)

    def register_trigger(self, event, fn):
        self._triggers[event] = fn

    def _trigger(self, event):
        if event in self._triggers:
            return self._triggers[event]()

    def symbol(self):
        return self._last_symbol

    def summary(self):
        return {
            "id": self._id,
            "t": self._time,
            "V": self.V,
            "E": self.E,
            "Θ": self.Θ,
            "ruptures": self._rupture_count,
            "last_symbol": self._last_symbol,
            "identity": self.identity
        }

    def last(self):
        return self.history[-1] if self.history else None

    def log(self):
        return self.history

    def rupture_log(self):
        return self.meta_ruptures
