<p align="center">
  <img src="https://raw.githubusercontent.com/heraclitus0/cognize/main/logo.png" width="180"/>
</p>

<h1 align="center">Cognize</h1>

<p align="center"><em>Programmable cognition for Python systems</em></p>

<p align="center">
  <a href="https://pypi.org/project/cognize"><img src="https://img.shields.io/pypi/v/cognize?color=blue&label=version" alt="Version"></a>
  <img src="https://img.shields.io/badge/python-3.8+-blue">
  <img src="https://img.shields.io/badge/status-beta-orange">
  <img src="https://img.shields.io/badge/license-Apache%202.0-black">
</p>

---

## Overview

**Cognize** is a lightweight cognition engine that tracks epistemic drift and enables rupture-aware reasoning in recursive systems.  
It models projection (`V`), reality (`R`), distortion (`∆`), misalignment memory (`E`), and rupture thresholds (`Θ`) — and supports programmable logic for collapse, realignment, and intervention.

---

## Features

- Drift-aware cognitive kernel (`EpistemicState`)
- Programmable rupture, realignment, and collapse policies
- Misalignment memory tracking with decay
- Symbolic cognition trace export (`.json`, `.csv`)
- Compatible with scalar or vector inputs
- DSL-ready via runtime injection (`inject_policy`)
- Lightweight, dependency-minimal, and test-backed

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

# Feed scalar signals
for R in [0.1, 0.3, 0.6, 0.8]:
    agent.receive(R)
    print(agent.summary())

# Inject custom rupture logic (optional)
from cognize.policies import collapse_soft_decay, realign_tanh, threshold_adaptive

agent.inject_policy(
    collapse=collapse_soft_decay,
    realign=realign_tanh,
    threshold=threshold_adaptive
)

# Run a new signal cycle
agent.receive(0.5)

# Get drift metrics
print(agent.drift_stats(window=3))

# Export cognition trace
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
  "event": "realign",
  "source": "default"
}
```

---

## License

Licensed under the [Apache 2.0 License](LICENSE).

---

© 2025 Pulikanti Sashi Bharadwaj  
All rights reserved.
