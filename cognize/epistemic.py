# Licensed under the Apache License, Version 2.0
"""
EpistemicState — Symbolic cognition engine for programmable epistemic drift tracking.
Implements Recursion Control Calculus (RCC) and Continuity Theory (CT) logic.

Symbols:
- ⊙ : Continuity Monad (fallback: CONTINUITY)
- ∆ : Distortion Function (fallback: DISTORTION)
- Θ : Rupture Threshold (fallback: THRESHOLD)
- E : Cumulative Misalignment (fallback: MISALIGNMENT)
- S̄ : Projected Divergence (fallback: DIVERGENCE)
- ∅ : No cognition received (fallback: EMPTY)
- ⚠ : Rupture detected (fallback: RUPTURE)
"""

__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache 2.0"
__version__ = "0.1.4"

import numpy as np
import uuid
from datetime import datetime


class EpistemicState:
    def __init__(self,
                 V0=0.0,
                 E0=0.0,
                 threshold=0.35,
                 realign_strength=0.3,
                 decay_rate=0.9,
                 identity=None,
                 log_history=True):
        self.V = float(V0)
        self.E = float(E0)
        self.Θ = float(threshold)
        self.k = float(realign_strength)
        self.decay_rate = float(decay_rate)

        self._threshold_fn = None
        self._realign_fn = None
        self._collapse_fn = None
        self._context_fn = None

        self.history = []
        self.meta_ruptures = []
        self._rupture_count = 0
        self._last_symbol = "∅"  # EMPTY
        self._triggers = {}
        self._time = 0
        self._id = str(uuid.uuid4())[:8]

        self.identity = identity or {}
        self._log = bool(log_history)
        self.event_log = []

    def receive(self, R, source="default"):
        R_val = self._resolve_reality(R)
        delta = abs(R_val - self.V)

        threshold = self._threshold_fn(self) if self._threshold_fn else self.Θ
        ruptured = delta > threshold
        self._last_symbol = "⚠" if ruptured else "⊙"

        if ruptured:
            if callable(self._collapse_fn):
                try:
                    self.V, self.E = self._collapse_fn(self)
                except Exception as e:
                    raise RuntimeError(f"Collapse function failed: {e}")
            else:
                self.V, self.E = 0.0, 0.0

            self._rupture_count += 1
            self.meta_ruptures.append({
                "time": self._time,
                "rupture_pressure": delta - threshold,
                "source": source
            })
            self._trigger("on_rupture")
        else:
            if callable(self._realign_fn):
                try:
                    self.V = self._realign_fn(self, R_val, delta)
                except Exception as e:
                    raise RuntimeError(f"Realign function failed: {e}")
            else:
                self.V += self.k * delta * (1 + self.E)
            self.E += 0.1 * delta
            self.E *= self.decay_rate

        if self._log:
            self.history.append({
                "t": self._time,
                "V": float(self.V),
                "R": float(R_val),
                "∆": float(delta),
                "Θ": float(threshold),
                "ruptured": ruptured,
                "symbol": self._last_symbol,
                "source": source
            })

        self._time += 1

    def rupture_risk(self):
        if not self.history:
            return None
        last = self.history[-1]
        return float(last["∆"] - last["Θ"])

    def should_intervene(self, margin=0.0):
        risk = self.rupture_risk()
        return risk is not None and risk > margin

    def intervene_if_ruptured(self, fallback_fn, margin=0.0):
        if self.should_intervene(margin):
            return fallback_fn()
        return None

    def reset(self):
        self.V = 0.0
        self.E = 0.0
        self._rupture_count = 0
        self._time = 0
        self._last_symbol = "∅"
        self.history.clear()
        self.meta_ruptures.clear()
        self.event_log.clear()

    def realign(self, R):
        R_val = self._resolve_reality(R)
        self.V = R_val
        self.E *= 0.5
        self._last_symbol = "⊙"

        if self._log:
            self.history.append({
                "t": self._time,
                "V": self.V,
                "R": R_val,
                "∆": 0.0,
                "Θ": self.Θ,
                "ruptured": False,
                "symbol": "⊙",
                "source": "manual_realign"
            })
        self._log_event("manual_realign", {"aligned_to": R_val})
        self._time += 1

    def inject_policy(self, threshold=None, realign=None, collapse=None):
        if threshold and not callable(threshold):
            raise TypeError("Threshold policy must be callable.")
        if realign and not callable(realign):
            raise TypeError("Realign policy must be callable.")
        if collapse and not callable(collapse):
            raise TypeError("Collapse policy must be callable.")

        self._threshold_fn = threshold
        self._realign_fn = realign
        self._collapse_fn = collapse
        self._log_event("policy_injected", {
            "threshold": bool(threshold),
            "realign": bool(realign),
            "collapse": bool(collapse)
        })

    def bind_context(self, fn):
        if not callable(fn):
            raise TypeError("Context function must be callable.")
        self._context_fn = fn
        self._log_event("context_bound", {"bound": True})

    def run_context(self, *args, **kwargs):
        if self._context_fn is None:
            raise ValueError("No context function bound.")
        result = self._context_fn(*args, **kwargs)
        self._log_event("context_executed", {"result": result})
        return result

    def register_trigger(self, event, fn):
        if not isinstance(event, str):
            raise TypeError("Event must be a string.")
        if not callable(fn):
            raise ValueError("Trigger must be callable.")
        self._triggers[event] = fn
        self._log_event("trigger_registered", {"event": event})

    def _trigger(self, event):
        if event in self._triggers:
            try:
                result = self._triggers[event]()
                self._log_event("trigger_invoked", {"event": event})
                return result
            except Exception as e:
                self._log_event("trigger_error", {"event": event, "error": str(e)})
                raise

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

    def drift_stats(self, window=10):
        deltas = [step['∆'] for step in self.history[-window:] if '∆' in step]
        if not deltas:
            return {}
        arr = np.array(deltas)
        return {
            "mean_drift": float(arr.mean()),
            "std_drift": float(arr.std()),
            "max_drift": float(arr.max()),
            "min_drift": float(arr.min())
        }

    def export_json(self, path):
        import json
        def sanitize(v):
            if isinstance(v, (np.integer, np.int64)): return int(v)
            if isinstance(v, (np.floating, np.float64)): return float(v)
            if isinstance(v, (np.ndarray, list)): return np.asarray(v).tolist()
            if isinstance(v, (str, bool, type(None))): return v
            try: return float(v)
            except: return str(v)

        try:
            safe_history = [{k: sanitize(v) for k, v in entry.items()} for entry in self.history]
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(safe_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to export JSON log to {path}: {e}")

    def export_csv(self, path):
        import csv
        if not self.history:
            return

        def sanitize(v):
            if isinstance(v, (np.integer, np.int64)): return int(v)
            if isinstance(v, (np.floating, np.float64)): return float(v)
            if isinstance(v, (np.ndarray, list)): return '|'.join(map(str, np.asarray(v)))
            return str(v)

        try:
            keys = sorted(set().union(*(entry.keys() for entry in self.history)))
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.history:
                    safe_row = {k: sanitize(row.get(k, '')) for k in keys}
                    writer.writerow(safe_row)
        except Exception as e:
            raise RuntimeError(f"Failed to export CSV log to {path}: {e}")

    def _log_event(self, event_type, details=None):
        sanitized = {}
        if details:
            for k, v in details.items():
                if isinstance(v, np.floating): sanitized[k] = float(v)
                elif isinstance(v, np.integer): sanitized[k] = int(v)
                else: sanitized[k] = v
        self.event_log.append({
            "event": event_type,
            "time": self._time,
            "timestamp": datetime.utcnow().isoformat(),
            "details": sanitized
        })

    def event_log_summary(self):
        return self.event_log

    def _resolve_reality(self, R):
        if isinstance(R, (int, float)):
            return float(R)
        elif isinstance(R, list):
            R = np.array(R)
        if isinstance(R, np.ndarray):
            if isinstance(self.V, (int, float)):
                self.V = np.zeros_like(R)
            return np.linalg.norm(R)
        raise ValueError("Reality must be float, list, or ndarray.")
