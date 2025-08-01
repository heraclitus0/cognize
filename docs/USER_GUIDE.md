# Cognize: User Guide

## What Is Cognize?

Cognize is a lightweight, programmable epistemic engine for tracking cognitive drift in Python systems. It models how internal beliefs (`V`) diverge from external signals (`R`) over time, maintaining a memory of misalignments (`E`) and determining whether rupture thresholds (`Θ`) are exceeded. It enables systems—agents, simulations, or APIs—to introspect, adapt, and realign dynamically.

This guide provides deep insights into how Cognize works, how to use it, and how to extend it.

---

## Cognition as Control

At its core, Cognize is not just a tracker—it's a **control system for epistemic state regulation**.

- **Projection (`V`)**: What the system *thinks* is true
- **Reality (`R`)**: What the environment *actually* signals
- **Distortion (`∆`)**: The magnitude of divergence between `V` and `R`
- **Threshold (`Θ`)**: Tolerance level before cognitive rupture occurs
- **Memory (`E`)**: How much past misalignment influences current sensitivity

---

## Installation

```bash
pip install cognize
```

---

## Cognitive Loop Flow

Each time `receive(R)` is called:

1. Compute `∆ = |R - V|`
2. Check: `∆ > Θ`?
   - Yes → Rupture: collapse state, log rupture
   - No → Realign: update `V` with a custom function (`⊙`)
3. Update misalignment memory `E` based on distortion and decay
4. Log the step, update time

---

## Default Logic

### Rupture
```python
if abs(R - V) > Θ:
    V, E = 0.0, 0.0
```

### Realignment
```python
V += k * delta * (1 + E)
E += 0.1 * delta
E *= decay_rate
```

---

## Customizing Logic

### Custom Threshold
```python
def threshold_adaptive(E, t):
    return 0.2 + 0.1 * E
```

### Custom Realign (⊙ operator)
```python
def realign_tanh(V, delta, E, k):
    return V + k * np.tanh(delta)
```

### Custom Collapse
```python
def collapse_soft_decay(R, V, E):
    return R * 0.1 + V * 0.9, E * 0.5
```

### Injecting
```python
agent.inject_policy(
    threshold=threshold_adaptive,
    realign=realign_tanh,
    collapse=collapse_soft_decay
)
```

---

## Drift Metrics & Symbolic State

### `drift_stats()`
Returns:
- mean drift
- std deviation
- max/min deltas over time

### `summary()`
Returns:
- current V, E, Θ
- total ruptures
- last event symbol (`rupture` or `realign`)
- identity and UUID

---

## Exporting Logs

### JSON Log
```python
agent.export_json("trace.json")
```

### CSV Log
```python
agent.export_csv("trace.csv")
```

Each step logs: `t`, `V`, `R`, `∆`, `Θ`, `ruptured`, `event`, `source`

---

## Advanced Usage

### Vector Reality (e.g., embeddings)
```python
import numpy as np
agent.receive(np.array([0.2, 0.4, 0.6]))
```

The vector norm is used for `∆`.

---

### Real-Time Interventions
```python
if agent.should_intervene(margin=0.05):
    agent.intervene_if_ruptured(lambda: trigger_realign())
```

---

### Cognitive Resets
```python
agent.reset()
```

Resets all internal state (`V`, `E`, rupture count, time, logs).

---

### Identity Assignment
```python
EpistemicState(identity={"module": "navigation_core", "agent_id": 17})
```

Stored and shown in `summary()`.

---

## Practical Patterns

| Pattern | Use Case |
|--------|----------|
| `decay_rate=0.0` | No memory — stateless response |
| High `k` (e.g. 0.9) | Fast realignment |
| Threshold as function of `E` | Memory-sensitized rupture |
| Realign using RMS / L2 logic | Multi-dimensional feedback |

---

## Tips & Design Insights

- **Don’t assume linear adaptation** — nonlinear realign functions give smoother or bounded updates.
- **Use logs as audit trails** — JSON traces are helpful for debugging LLM drift or simulation cycles.
- **Model decay carefully** — high `E` decay makes rupture less sensitive to history.
- **Symbolic abstraction is intentional** — Cognize avoids leaking core math; it enables programmable representations.

---

## FAQ

### Is this a filter?
No. Cognize is an epistemic control system, not a signal smoother. It models cognitive drift and intervention logic.

### Does rupture always reset to zero?
Only by default. You can override with soft decay, target-driven resets, or symbolic tagging.

### Can I model memory fatigue?
Yes — adjust how `E` decays or write decay rules that increase volatility after repeated misalignments.

---

## License

Apache 2.0

---

© 2025 Pulikanti Sashi Bharadwaj
