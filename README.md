<p align="center">
  <img src="https://raw.githubusercontent.com/heraclitus0/cognize/main/assets/logo.png" width="160"/>
</p>

<h1 align="center">Cognize</h1>
<p align="center"><em>Programmable cognition for Python systems</em></p>

<p align="center">
  <a href="https://pypi.org/project/cognize"><img src="https://img.shields.io/pypi/v/cognize?color=blue&label=version" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.8+-blue">
  <img src="https://img.shields.io/badge/status-beta-violet">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <a href="https://pepy.tech/project/cognize"><img src="https://static.pepy.tech/badge/cognize" alt="Downloads"></a>
  <a href="https://doi.org/10.5281/zenodo.17042860"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17042860.svg" alt="DOI"></a>
</p>

---

## Overview

**Cognize** is a lightweight cognition engine for Python.  
It tracks a system’s **belief** (`V`) against **reality** (`R`), accumulates **misalignment memory** (`E`), and triggers **rupture** when drift exceeds a threshold (`Θ`).  

It’s programmable at runtime — inject your own threshold, realignment, and collapse logic, or use the included safe presets.  

**Use cases:**  
- **Anomaly detection & drift-aware pipelines**  
- **Cognitive & adaptive agents** (systems that self-correct against misalignment)  
- **Simulations of epistemic systems** (belief, memory, rupture dynamics)  
- **Metacognitive mechanics** (self-monitoring, policy evolution, reflective control)  

---

## Features

- **Epistemic kernel**: `EpistemicState` with scalar & vector support  
- **Programmable policies**: custom `threshold`, `realign`, `collapse` functions  
- **Perception adapter**: fuse text/image/sensor inputs into a normalized vector  
- **Meta-policy selection**: `PolicyManager` with shadow evaluation & safe promotion  
- **Explainability**: rolling logs, `explain_last()`, JSON/CSV export  
- **Lightweight core**: NumPy-only dependency, optional visualization  

---

## Install

```bash
pip install cognize
```

---

## Core Concepts

| Symbol | Meaning               |
|:------:|-----------------------|
| `V`    | Belief / Projection   |
| `R`    | Reality signal        |
| `∆`    | Distortion (`R−V`)    |
| `Θ`    | Rupture threshold     |
| `E`    | Misalignment memory   |
| `⊙`    | Realignment operator  |

---

## Quick Start

```python
from cognize import EpistemicState
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay

state = EpistemicState(V0=0.5, threshold=0.35, realign_strength=0.3)
state.inject_policy(
    threshold=threshold_adaptive,
    realign=realign_tanh,
    collapse=collapse_soft_decay,
)

for r in [0.1, 0.3, 0.7, 0.9]:
    state.receive(r)

print(state.explain_last())  # human-readable step summary
print(state.summary())       # compact state snapshot
```

---

## Multi-modal Example

```python
import numpy as np
from cognize import EpistemicState, Perception

def toy_text_encoder(s: str) -> np.ndarray:
    return np.array([len(s), s.count(" "), s.count("a"), 1.0], dtype=float)

P = Perception(text_encoder=toy_text_encoder)
state = EpistemicState(V0=np.zeros(4), perception=P)

state.receive({"text": "hello world"})
print(state.last())
```

Supports multiple modalities in one dict, e.g.:  
```python
{"text": "...", "image": img, "sensor": {...}}
```

---

## Meta-policy Selection

```python
from cognize import EpistemicState, PolicyManager, PolicyMemory, ShadowRunner, SAFE_SPECS

state = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.3)
state.policy_manager = PolicyManager(
    base_specs=SAFE_SPECS,  # conservative / cautious / adoptive
    memory=PolicyMemory(),
    shadow=ShadowRunner(),
    epsilon=0.15,
    promote_margin=1.03,
    cooldown_steps=30
)

for r in [0.2, 0.4, 0.45, 0.5, 0.6, 0.65, 0.62, 0.58, 0.7, 0.72, 0.69, 0.75, 0.8]:
    state.receive(r)

print(state.event_log_summary()[-3:])
```

---

## API Surface

```python
from cognize import (
    EpistemicState, EpistemicGraph, Perception,
    PolicyManager, PolicySpec, PolicyMemory, ShadowRunner, SAFE_SPECS,
    POLICY_REGISTRY,
    make_simple_state,
)
```

- `EpistemicState`: epistemic kernel (receive evidence, track drift, export traces)  
- `Perception`: adapter for multi-modal inputs  
- `PolicyManager`: ε-greedy with shadow evaluation & promotion  
- `SAFE_SPECS`: conservative / cautious / adoptive presets  
- `EpistemicGraph`: networks of interacting states  

[User Guide](https://github.com/heraclitus0/cognize/blob/main/docs/USER_GUIDE.md)

---

## Citation

If you use **Cognize**, please cite:

```bibtex
@software{pulikanti2025cognize,
  author       = {Pulikanti, Sashi Bharadwaj},
  title        = {Cognize: Programmable cognition for Python systems},
  version      = {0.1.7},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17042860},
  url          = {https://doi.org/10.5281/zenodo.17042860}
}
```

Full citation formats (APA, MLA, etc.) are available via the DOI badge.

---

## License

Licensed under the **Apache License 2.0**.  
© 2025 Pulikanti Sashi Bharadwaj
