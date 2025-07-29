
"""
Cognize Simulation Module

This module provides a lightweight simulation engine to model misalignment between internal projections
and external inputs in dynamic agents. It exposes tools for realignment, rupture detection, and recovery
through configurable signal-response loops.

Inspired by epistemic simulation theory, but decoupled from any specific doctrine.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_signal(mode="shock", steps=50, seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)

    R = kwargs.get("start", 0.0)
    freq = kwargs.get("freq", 0.1)
    noise = kwargs.get("noise", 0.05)
    shock_at = kwargs.get("shock_at", steps // 2)
    shock_magnitude = kwargs.get("shock_magnitude", 2.0)
    const_value = kwargs.get("value", 0.0)

    for t in range(steps):
        if mode == "random_walk":
            R += np.random.normal(0, noise)
        elif mode == "oscillate":
            R = np.sin(t * freq)
        elif mode == "shock":
            R += shock_magnitude if t == shock_at else np.random.normal(0, noise)
        elif mode == "constant":
            R = const_value
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        yield R

def run_simulation(
    steps=50,
    signal_type="shock",
    realign_strength=0.3,
    threshold_base=0.35,
    noise_level=0.025
):
    V = 0.0
    E = 0.0
    trace = []

    signal = generate_signal(mode=signal_type, steps=steps)

    for t, R in enumerate(signal):
        delta = abs(R - V)
        theta = threshold_base + E * 0.05 + np.random.normal(0, noise_level)

        ruptured = delta > theta

        if ruptured:
            V = 0.0
            E = 0.0
            collapse_type = "reset"
        else:
            V += realign_strength * delta * (1 + E)
            E += 0.1 * delta
            collapse_type = None

        trace.append({
            "t": t,
            "V": V,
            "R": R,
            "∆": delta,
            "Θ": theta,
            "ruptured": ruptured,
            "collapse_type": collapse_type
        })

    return trace

def plot_trace(trace):
    t = [s["t"] for s in trace]
    V = [s["V"] for s in trace]
    R = [s["R"] for s in trace]
    delta = [s["∆"] for s in trace]
    theta = [s["Θ"] for s in trace]
    ruptured = [s["ruptured"] for s in trace]

    plt.figure(figsize=(12, 6))
    plt.plot(t, V, label="Projection V(t)", linewidth=2)
    plt.plot(t, R, label="Reality R(t)", linestyle="--")
    plt.plot(t, delta, label="Distortion ∆(t)", linestyle="-.")
    plt.plot(t, theta, label="Threshold Θ(t)", linestyle=":")

    rupture_t = [tt for i, tt in enumerate(t) if ruptured[i]]
    rupture_V = [vv for i, vv in enumerate(V) if ruptured[i]]
    plt.scatter(rupture_t, rupture_V, color="red", label="Ruptures")

    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Cognitive Drift and Rupture Trace")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
