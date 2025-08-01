# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.

"""
EpistemicState — Symbolic cognition engine for programmable epistemic drift tracking.
Part of the Cognize project (https://pypi.org/project/cognize/).
"""

__author__ = "Pulikanti Sashi Bharadwaj"
__license__ = "Apache 2.0"
__version__ = "0.1.3"

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
    """
    Initializes the epistemic state.

    Args:
        V0 (float): Initial projection (belief).
        E0 (float): Initial misalignment memory.
        threshold (float): Static rupture threshold if no dynamic logic injected.
        realign_strength (float): Base gain for default realignment.
        decay_rate (float): Memory decay factor after each step.
        identity (dict): Optional identity metadata.
        log_history (bool): Whether to keep a step-by-step cognition trace.
    """

    # Core epistemic fields
    self.V = float(V0)
    self.E = float(E0)
    self.Θ = float(threshold)
    self.k = float(realign_strength)
    self.decay_rate = float(decay_rate)

    # Hooks for dynamic logic
    self._threshold_fn = None
    self._realign_fn = None
    self._collapse_fn = None
    self._context_fn = None

    # Internal cognitive state
    self.history = []
    self.meta_ruptures = []
    self._rupture_count = 0
    self._last_symbol = "∅"
    self._triggers = {}
    self._time = 0
    self._id = str(uuid.uuid4())[:8]

    # Agent metadata
    self.identity = identity or {}

    # Logging toggle
    self._log = bool(log_history)

def receive(self, R, source="default"):
    """
    Ingests a new reality signal and updates internal cognitive state.

    Args:
        R (float, list, or np.ndarray): Incoming signal or vector representing external reality.
        source (str): Optional label for origin of signal (for audit trace).
    """

    R_val = self._resolve_reality(R)
    delta = abs(R_val - self.V)

    # Dynamic or static rupture threshold
    threshold = self._threshold_fn(self) if self._threshold_fn else self.Θ
    ruptured = delta > threshold
    self._last_symbol = "⚠" if ruptured else "⊙"

    if ruptured:
        # Collapse logic (injected or default)
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
        # Realignment logic (injected or default)
        if callable(self._realign_fn):
            try:
                self.V = self._realign_fn(self, R_val, delta)
            except Exception as e:
                raise RuntimeError(f"Realign function failed: {e}")
        else:
            self.V += self.k * delta * (1 + self.E)

        self.E += 0.1 * delta
        self.E *= self.decay_rate

    # Log cognitive state
    if self._log:
        self.history.append({
            "t": self._time,
            "V": float(self.V),
            "R": float(R_val),
            "delta": float(delta),
            "Θ": float(threshold),
            "ruptured": ruptured,
            "symbol": self._last_symbol,
            "source": source
        })

    self._time += 1

def rupture_risk(self):
    """
    Returns the current epistemic rupture pressure (∆ - Θ).

    This is a real-time metric of how close the current signal is to exceeding
    the cognitive rupture threshold. Useful for agents that pre-emptively 
    trigger interventions or fallback behavior before rupture occurs.

    Returns:
        float or None: Positive value if near rupture, negative if stable. 
                       None if no signal has been received yet.
    """
    if not self.history:
        return None

    last_delta = self.history[-1].get("delta", None)
    last_threshold = self.history[-1].get("Θ", self.Θ)

    if last_delta is None:
        return None

    return float(last_delta - last_threshold)


def should_intervene(self, margin=0.0):
    """
    Checks whether current cognitive drift exceeds the rupture threshold plus a margin.
    
    This function allows the system to anticipate potential rupture before it happens,
    enabling proactive fallback logic or stabilizing behavior.

    Args:
        margin (float): Optional buffer to increase sensitivity. 
                        If margin > 0, intervenes earlier than actual rupture.

    Returns:
        bool: True if current rupture pressure exceeds the margin, else False.
    """
    risk = self.rupture_risk()
    
    if risk is None:
        return False  # No signal yet; nothing to assess

    return risk > margin

def intervene_if_ruptured(self, fallback_fn, margin=0.0):
    """
    Executes a fallback intervention function if the current epistemic state
    is at risk of rupture (drift exceeds threshold + margin).

    Args:
        fallback_fn (callable): A function to be called as an intervention.
        margin (float): Optional early warning buffer. Default is 0.0.

    Returns:
        Any: The return value of fallback_fn if triggered, else None.
    """
    if self.should_intervene(margin):
        return fallback_fn()
    return None
def reset(self):
    """
    Fully resets the internal epistemic state of the agent.

    This clears:
    - projection (`V`)
    - misalignment memory (`E`)
    - rupture count
    - step history (`log`)
    - rupture log (`meta_ruptures`)
    - time index (`t`)
    - last event symbol

    Does not reset:
    - injected policies
    - identity metadata
    - context or triggers
    """
    self.V = 0.0
    self.E = 0.0
    self._rupture_count = 0
    self._time = 0
    self._last_symbol = "∅"
    self.history.clear()
    self.meta_ruptures.clear()

    # Optional: reset event log if exists
    if hasattr(self, 'event_log'):
        self.event_log.clear()

def realign(self, R):
    """
    Manually forces projection `V` to align with a given reality signal `R`.

    - Replaces the current `V` with the resolved value of `R`
    - Applies decay to memory `E` (simulates partial forgetfulness)
    - Logs the realignment event as a non-rupture (`symbol = ⊙`)
    - Increments the internal time counter

    Use when:
    - You want to force a reset to reality
    - Realignment is contextually required (e.g., after manual override)
    """
    R_val = self._resolve_reality(R)
    self.V = R_val
    self.E *= 0.5  # Simulate memory decay
    self._last_symbol = "⊙"

    if self._log:
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

    self._log_event("manual_realign", details={"aligned_to": R_val})
    self._time += 1


def inject_policy(self, threshold=None, realign=None, collapse=None):
    """
    Injects user-defined logic into the epistemic engine for:
    - rupture threshold adaptation (`threshold(self) -> float`)
    - realignment kernel (`realign(self, R, delta) -> V`)
    - collapse handler on rupture (`collapse(self) -> (V, E)`)

    These override default logic and allow full customization of:
    - when rupture occurs
    - how V responds to ∆
    - what happens on rupture (reset, decay, adapt)

    Parameters:
        threshold (callable): Function that takes `self` and returns new threshold.
        realign   (callable): Function taking `(self, R, delta)` and returning updated V.
        collapse  (callable): Function taking `self` and returning new (V, E).

    Example:
    >>> agent.inject_policy(
    ...     threshold=lambda self: 0.3 + 0.1 * self.E,
    ...     realign=lambda self, R, d: self.V + d * 0.5,
    ...     collapse=lambda self: (0.0, 0.0)
    ... )
    """
    if threshold is not None and not callable(threshold):
        raise TypeError("Threshold policy must be a callable.")
    if realign is not None and not callable(realign):
        raise TypeError("Realign policy must be a callable.")
    if collapse is not None and not callable(collapse):
        raise TypeError("Collapse policy must be a callable.")

    self._threshold_fn = threshold
    self._realign_fn = realign
    self._collapse_fn = collapse

    self._log_event("policy_injected", {
        "threshold": bool(threshold),
        "realign": bool(realign),
        "collapse": bool(collapse)
    })


def bind_context(self, fn):
    """
    Binds an external callable (e.g., context evaluator or environment responder)
    to the agent's cognition loop.

    This enables:
    - Injecting external evaluation logic
    - Hooking dynamic environment conditions
    - Integrating LLMs, APIs, or simulation conditions

    The bound function can later be invoked with `run_context(...)`.

    Parameters:
        fn (callable): A function to bind. It can accept any args/kwargs.

    Example:
    >>> def context_fn(agent_state):
    ...     return agent_state.V > 0.5

    >>> agent.bind_context(context_fn)
    """
    if not callable(fn):
        raise TypeError("Context function must be callable.")
    self._context_fn = fn
    self._log_event("context_bound", {"bound": True})


def run_context(self, *args, **kwargs):
    """
    Executes the external context function previously bound via `bind_context(...)`.

    This allows the agent to:
    - Dynamically query external inputs or feedback loops
    - Adapt decisions based on current state or external environment
    - Trigger realignment, rupture, or symbolic labeling via indirect logic

    Parameters:
        *args, **kwargs: Optional arguments passed to the bound context function.

    Returns:
        Output of the context function.

    Raises:
        ValueError: If no context function has been bound.

    Example:
    >>> agent.bind_context(lambda: "safe" if agent.V < 0.4 else "drift")
    >>> agent.run_context()
    "safe"
    """
    if self._context_fn is None:
        raise ValueError("No context function bound to run_context(). Use bind_context(fn) first.")

    result = self._context_fn(*args, **kwargs)
    self._log_event("context_executed", {"result": result})
    return result
def register_trigger(self, event, fn):
    """
    Registers a symbolic trigger for a given event type.

    This allows external systems to inject behavior when rupture or symbolic
    events occur — e.g., re-training, fallback strategies, logging, or rerouting.

    Parameters:
        event (str): Event name to hook into. Typical examples:
                     - "on_rupture"
                     - "on_realign"
                     - "on_reset"
                     - "on_external_override"

        fn (callable): A function to be called when the event occurs.
                       Should take no parameters or handle via closure.

    Example:
    >>> def alert(): print("⚠ Rupture detected!")
    >>> agent.register_trigger("on_rupture", alert)
    """
    if not isinstance(event, str):
        raise TypeError("Event name must be a string.")

    if not callable(fn):
        raise ValueError("Trigger must be a callable function.")

    self._triggers[event] = fn
    self._log_event("trigger_registered", {"event": event, "function": repr(fn)})

def _trigger(self, event):
    """
    Internally invokes a registered trigger for a given symbolic event.

    This is called inside critical functions (e.g., rupture events) to allow
    external behaviors (notifications, re-training, symbolic logging, etc.).

    Parameters:
        event (str): The name of the event to trigger (e.g., "on_rupture")

    Returns:
        The result of the triggered function (if any), else None.

    Example:
        self._trigger("on_rupture")
    """
    if event in self._triggers:
        try:
            result = self._triggers[event]()
            self._log_event("trigger_invoked", {"event": event})
            return result
        except Exception as e:
            self._log_event("trigger_error", {
                "event": event,
                "error": str(e)
            })
            raise RuntimeError(f"Error during trigger '{event}': {e}")


def symbol(self):
    """
    Returns the symbolic status of the last step:
    - '⊙' indicates stable cognition (realignment without rupture)
    - '⚠' indicates rupture (epistemic collapse)
    - '∅' indicates no input has been received yet

    Returns:
        str: One of '⊙', '⚠', or '∅'
    """
    return self._last_symbol
def summary(self):
    """
    Returns a snapshot summary of the agent's current epistemic state.

    Returns:
        dict: {
            'id': str,             # Unique agent/session ID
            't': int,              # Time step
            'V': float or np.array,# Current belief or projection
            'E': float,            # Misalignment memory
            'Θ': float,            # Default rupture threshold (if no dynamic fn injected)
            'ruptures': int,       # Total number of rupture events
            'last_symbol': str,    # Last symbolic status: '⊙', '⚠', or '∅'
            'identity': dict       # Optional custom metadata (module, agent id, etc.)
        }
    """
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
    """
    Returns the most recent cognition log entry from the history.

    Returns:
        dict or None: A dictionary representing the latest recorded step,
                      or None if no steps have been recorded yet.
    """
    return self.history[-1] if self.history else None

def log(self):
    """
    Returns the full cognition trace recorded during the agent's lifetime.

    Returns:
        list: A list of dictionaries, each representing a timestep's cognition event.
    """
    return self.history


def rupture_log(self):
    """
    Returns a list of all rupture events encountered by the agent.

    Returns:
        list: Each item is a dictionary containing rupture details such as time, pressure, and source.
    """
    return self.meta_ruptures

def drift_stats(self, window=10):
    """
    Computes rolling statistics on the last `window` steps of cognitive distortion (∆).

    Args:
        window (int): Number of past steps to include in the analysis.

    Returns:
        dict: {
            "mean_drift": float,
            "std_drift": float,
            "max_drift": float,
            "min_drift": float
        }
        or empty dict if insufficient data.
    """
    deltas = [step['delta'] for step in self.history[-window:] if 'delta' in step]
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
    """
    Safely exports history log to a JSON file, with fallbacks and encoding guards.
    """
    import json

    def sanitize(value):
        if isinstance(value, (np.integer, np.int64, np.int32)): return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)): return float(value)
        if isinstance(value, (np.ndarray, list)): return np.asarray(value).tolist()
        if isinstance(value, (str, bool, type(None))): return value
        try:
            return float(value)
        except:
            return str(value)  # Fallback catch-all

    try:
        safe_history = [
            {k: sanitize(v) for k, v in entry.items()} for entry in self.history
        ]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(safe_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to export JSON log to {path}: {e}")

def export_csv(self, path):
    """
    Safely exports history log to a CSV file.
    Handles NumPy types and missing fields.
    """
    import csv

    if not self.history:
        return

    def sanitize(value):
        if isinstance(value, (np.integer, np.int64, np.int32)): return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)): return float(value)
        if isinstance(value, (np.ndarray, list)): return '|'.join(map(str, np.asarray(value)))
        return str(value)

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
    """Logs symbolic events such as ruptures or manual realignments."""
    if not hasattr(self, 'event_log') or self.event_log is None:
        self.event_log = []

    sanitized_details = details or {}
    try:
        # Clean numpy types for safety
        sanitized_details = {
            k: (
                float(v) if isinstance(v, np.floating) else
                int(v) if isinstance(v, np.integer) else
                v
            )
            for k, v in sanitized_details.items()
        }
    except Exception:
        pass  # fallback if sanitization fails

    self.event_log.append({
        "event": event_type,
        "time": self._time,
        "timestamp": datetime.utcnow().isoformat(),
        "details": sanitized_details
    })


def event_log_summary(self):
    """
    Returns a list of symbolic events that occurred during runtime,
    including ruptures, manual interventions, and triggered callbacks.
    """
    if not hasattr(self, 'event_log') or self.event_log is None:
        return []
    return self.event_log

def _resolve_reality(self, R):
    """
    Resolves the reality input `R` into a scalar value.

    Supports:
    - Scalar input (int, float)
    - List input (converted to NumPy array, uses L2 norm)
    - NumPy array input (uses L2 norm)
    
    If the projection `V` is scalar but `R` is a vector, `V` is promoted to a zero-vector to match.

    Raises:
        ValueError: if `R` is not a supported type.
    """
    if isinstance(R, (int, float)):
        return float(R)
    
    elif isinstance(R, list):
        R = np.array(R)

    if isinstance(R, np.ndarray):
        if isinstance(self.V, (int, float)):
            self.V = np.zeros_like(R)
        return np.linalg.norm(R)

    raise ValueError("Reality must be a float, list, or NumPy ndarray.")
