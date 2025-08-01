<p align="center">
  <img src="https://raw.githubusercontent.com/heraclitus0/cognize/main/assets/logo.png" width="180"/>
</p>

<h1 align="center">Cognize</h1>

<p align="center"><em>Programmable cognition for Python systems</em></p>

<p align="center">
  <a href="https://pypi.org/project/cognize"><img src="https://img.shields.io/pypi/v/cognize?color=blue&label=version" alt="Version"></a>
  <img src="https://img.shields.io/badge/python-3.8+-blue">
  <img src="https://img.shields.io/badge/status-beta-orange">
  <img src="https://img.shields.io/badge/license-Apache%202.0-red">
</p>

---

## Overview

**Cognize** is a lightweight cognition engine for Python systems.  
It tracks belief (`V`) vs. reality (`R`), manages misalignment memory (`E`), and detects symbolic rupture (`Θ`).  
Now supports runtime injection of programmable logic for collapse, realignment, and adaptive thresholds.

Built for agents, simulations, filters, and symbolic drift-aware systems.

---

## Features

- Cognitive projection engine (`EpistemicState`)
- Drift tracking with misalignment memory
- Programmable `inject_policy(...)` support
- Prebuilt logic in `cognize.policies` (collapse, realign, threshold)
- Vector-compatible input support
- Trace export (`.json`, `.csv`) for audit or training
- Lightweight, domain-agnostic, DSL-ready

---

## Installation

```bash
pip install cognize
```

---

## Core Concepts

| Symbol | Meaning             |
|--------|---------------------|
| `V`    | Belief / Projection |
| `R`    | Reality Signal      |
| `∆`    | Distortion          |
| `Θ`    | Rupture Threshold   |
| `E`    | Misalignment Memory |

---

## Quick Usage

```python
from cognize import EpistemicState

# Initialize agent
agent = EpistemicState(V0=0.0, threshold=0.35)

# Feed signals
for R in [0.1, 0.3, 0.6, 0.8]:
    agent.receive(R)
    print(agent.summary())
```

---

## Programmable Injection (v0.1.2+)

```python
from cognize.policies import (
    collapse_soft_decay_fn,
    realign_tanh_fn,
    threshold_adaptive_fn
)

agent.inject_policy(
    collapse=collapse_soft_decay_fn,
    realign=realign_tanh_fn,
    threshold=threshold_adaptive_fn
)

agent.receive(0.5)
print(agent.summary())
```

---

## Exporting Trace

```python
agent.export_json("trace.json")
agent.export_csv("trace.csv")
```

---

## Example Output

```json
{
  "t": 2,
  "V": 0.41,
  "R": 0.6,
  "delta": 0.19,
  "Θ": 0.35,
  "ruptured": false,
  "symbol": "⊙",
  "source": "default"
}
```

---

[Full Cognize User Guide](https://github.com/heraclitus0/cognize/blob/main/docs/USER_GUIDE.md)

---

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

© 2025 Pulikanti Sashi Bharadwaj  
All rights reserved.
