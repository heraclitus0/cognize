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

**Cognize** is a lightweight cognition engine that tracks a system’s **belief** (`V`) against **reality** (`R`), accumulates **misalignment memory** (`E`), and triggers **rupture** when drift exceeds a threshold (`Θ`).  
It’s programmable at runtime—inject your own threshold, realignment, and collapse logic, or use the included safe presets.

Built for agents, simulations, filters, anomaly detection, and drift‑aware pipelines.

---

## Features

- **Epistemic kernel**: `EpistemicState` with scalar & vector support
- **Programmable policies**: inject custom `threshold`, `realign`, `collapse` functions
- **Perception adapter**: normalize text/image/sensor inputs into a fused vector
- **Meta‑policy selection**: `PolicyManager` with shadow evaluation & safe promotion
- **Explainability**: rolling logs, `explain_last()`, JSON/CSV export
- **Tiny core**: hard dependency only on NumPy; visualization is optional

---

## Install

```bash
pip install cognize
```

---

## Core Concepts

| Symbol | Meaning                      |
|:------:|------------------------------|
| `V`    | Belief / Projection          |
| `R`    | Reality signal               |
| `∆`    | Distortion (`R−V`)           |
| `Θ`    | Rupture threshold            |
| `E`    | Misalignment memory          |
| `⊙`    | Realignment operator         |

---

## Quick start (scalar)

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

# Export full trace
state.export_json("run.json")
state.export_csv("run.csv")
```

---

## Multi‑modal with `Perception`

Bring your own encoders and fuse modalities into a single evidence vector.

```python
import numpy as np
from cognize import EpistemicState, Perception

# Toy 4‑D text embedding
def toy_text_encoder(s: str) -> np.ndarray:
    return np.array([len(s), s.count(" "), s.count("a"), 1.0], dtype=float)

P = Perception(text_encoder=toy_text_encoder)  # defaults handle fusion & normalization
state = EpistemicState(V0=np.zeros(4), perception=P)

state.receive({"text": "hello world"})  # dict -> Perception -> vector
print(state.last())
```

You can pass multiple modalities in one dict, e.g. `{"text": "...", "image": img, "sensor": {...}}`.  
See `cognize/perception.py` for fusion, normalization, modality weights, and explain hooks.

---

## Meta‑policy selection (safe presets)

Use the built‑in `PolicyManager` to evaluate policy candidates in a **shadow** run and promote them when they outperform the current behavior.

```python
from cognize import EpistemicState, PolicyManager, PolicyMemory, ShadowRunner, SAFE_SPECS

state = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.3, rng_seed=42)
state.policy_manager = PolicyManager(
    base_specs=SAFE_SPECS,         # conservative / cautious / adoptive
    memory=PolicyMemory(),
    shadow=ShadowRunner(),
    epsilon=0.15,                  # exploration rate
    promote_margin=1.03,           # must beat baseline by 3%
    cooldown_steps=30
)

for r in [0.2, 0.4, 0.45, 0.5, 0.6, 0.65, 0.62, 0.58, 0.7, 0.72, 0.69, 0.75, 0.8]:
    state.receive(r)

print(state.event_log_summary()[-3:])   # see policy promotions/evolution events
```

Want bounded on‑policy evolution?

```python
state.enable_auto_evolution(
    param_space={
        "conservative": {"k": (0.1, 0.3), "Θ": (0.2, 0.6)},
        "cautious":     {"k": (0.1, 0.25), "Θ": (0.2, 0.6)},
        "adoptive":     {"k": (0.2, 0.35), "Θ": (0.2, 0.6)},
    },
    every=30, rate=1.0, margin=1.02
)
```

---

## API surface

```python
from cognize import (
    EpistemicState, EpistemicGraph, Perception,
    PolicyManager, PolicySpec, PolicyMemory, ShadowRunner, SAFE_SPECS,
    POLICY_REGISTRY,          # prebuilt policy registry (threshold/realign/collapse)
    make_simple_state,        # convenience factory
)
```

- `EpistemicState`: kernel (receive evidence, log steps, export traces)
- `Perception`: vector-fused adapter for dict inputs
- `PolicyManager`: ε‑greedy + shadow evaluation + safe promotion
- `SAFE_SPECS`: conservative / cautious / adoptive presets
- `EpistemicGraph`: multi‑state graphs (networks of interacting states)

**User Guide:** https://github.com/heraclitus0/cognize/blob/main/docs/USER_GUIDE.md

---


## License

Licensed under the **Apache License 2.0**.  
© 2025 Pulikanti Sashi Bharadwaj
