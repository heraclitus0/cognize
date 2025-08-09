<p align="center">
  <img src="https://raw.githubusercontent.com/heraclitus0/cognize/main/assets/logo.png" width="180"/>
</p>

<h1 align="center">Cognize</h1>
<p align="center"><em>Programmable cognition for Python systems</em></p>

<p align="center">
  <a href="https://pypi.org/project/cognize"><img src="https://img.shields.io/pypi/v/cognize?color=blue&label=version" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.8+-blue">
  <img src="https://img.shields.io/badge/status-beta-orange">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <a href="https://pepy.tech/project/cognize"><img src="https://static.pepy.tech/badge/cognize" alt="Downloads"></a>
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
import json
import numpy as np
from cognize import EpistemicState
from cognize.policies import collapse_soft_decay_fn, realign_tanh_fn, threshold_adaptive_fn

def to_py(x):
    if isinstance(x, (np.bool_,)):        return bool(x)
    if isinstance(x, (np.integer,)):      return int(x)
    if isinstance(x, (np.floating,)):     return float(x)
    if isinstance(x, (list, tuple)):      return [to_py(v) for v in x]
    if isinstance(x, dict):               return {k: to_py(v) for k, v in x.items()}
    return x

robot = EpistemicState(V0=0.5)
robot.inject_policy(
    collapse=collapse_soft_decay_fn,
    realign=realign_tanh_fn,
    threshold=threshold_adaptive_fn,
)

sensor_readings = [0.1, 0.3, 0.7, 0.9]
for reading in sensor_readings:
    robot.receive(reading)

print(json.dumps(to_py(robot.log()), indent=2, ensure_ascii=False))

```


## Example Output

```jason
[
  {
    "t": 0,
    "V": 0.25,
    "R": 0.1,
    "∆": 0.4,
    "Θ": 0.35,
    "ruptured": true,
    "symbol": "⚠",
    "source": "default"
  },
  {
    "t": 1,
    "V": 0.26499887510124076,
    "R": 0.3,
    "∆": 0.04999999999999999,
    "Θ": 0.35,
    "ruptured": false,
    "symbol": "⊙",
    "source": "default"
  },
  {
    "t": 2,
    "V": 0.13249943755062038,
    "R": 0.7,
    "∆": 0.4350011248987592,
    "Θ": 0.35022499999999995,
    "ruptured": true,
    "symbol": "⚠",
    "source": "default"
  },
  {
    "t": 3,
    "V": 0.06624971877531019,
    "R": 0.9,
    "∆": 0.7675005624493796,
    "Θ": 0.3500675,
    "ruptured": true,
    "symbol": "⚠",
    "source": "default"
  }
]
```



---

[Full Cognize User Guide](https://github.com/heraclitus0/cognize/blob/main/docs/USER_GUIDE.md)

---

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

© 2025 Pulikanti Sashi Bharadwaj  
All rights reserved.
