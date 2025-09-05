<p align="center">
  <img src="https://raw.githubusercontent.com/heraclitus0/cognize/main/assets/logo.png" width="160"/>
</p>

<h1 align="center">Cognize</h1>
<p align="center"><em>Programmable cognition for Python systems</em></p>

<p align="center">
  <a href="https://pypi.org/project/cognize"><img src="https://img.shields.io/pypi/v/cognize?color=blue&label=version" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-beta-orange" alt="Status: beta">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <a href="https://pepy.tech/project/cognize"><img src="https://static.pepy.tech/badge/cognize" alt="Downloads"></a>
  <a href="https://doi.org/10.5281/zenodo.17042859"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17042859.svg" alt="DOI"></a>
</p>

---

## Overview

**Cognize** is a lightweight cognition engine for Python.  
It tracks a system’s **belief** (`V`) against **reality** (`R`), accumulates **misalignment memory** (`E`), and triggers **rupture** when drift exceeds a threshold (`Θ`).

It’s programmable at runtime — inject your own threshold, realignment, and collapse logic, or use the included safe presets.

---

## Features

- **Epistemic kernel** — `EpistemicState` (scalar & vector), tracking `V, R, Δ, Θ, E`, with rupture and step-capped updates.
- **Programmable policies** — inject custom `threshold`, `realign`, `collapse` functions or use safe presets (`cognize.policies`).
- **Perception adapter** — `Perception` fuses text/image/sensor inputs into a normalized vector; bring your own encoders.
- **Meta-policy selection** — `PolicyManager` with shadow evaluation, ε-greedy exploration, and safe promotion (`SAFE_SPECS`).
- **Epistemic graphs** — `EpistemicGraph` / `EpistemicProgrammableGraph` orchestrate states via directed, decay/cooldown-aware links with programmable edges (gate → influence → magnitude → target(slice) → nudge → damp).
- **Meta-learning bounds** — `ParamRange`, `ParamSpace`, `enable_evolution()` and `enable_dynamic_evolution()` for bounded/static or provider-driven evolution.
- **Safety & telemetry by design** — step caps, oscillation damping, cooldowns; per-edge influence logs, cascade traces, `explain_last()`, CSV/JSON export.
- **Ergonomic helpers** — `make_simple_state`, `make_graph`, `demo_text_encoder` for fast setup.
- **Lightweight core** — NumPy-only dependency; optional viz/dev extras.

---

## Use Cases

- **Drift & anomaly detection (streaming)** — compute `Δ, E, Θ`; trigger ruptures; emit CSV/JSON telemetry for dashboards.
- **Continual-learning guardrails** — under non-stationarity, apply reversible cooling (`Θ↑`, `k↓` / LR↓) to reduce catastrophic forgetting.
- **Modulation for NNs (no retrain)** — runtime, slice-level nudges (attention logits, LayerNorm γ, MoE gates, temperatures) with caps & logs.
- **Multimodal arbitration (explainable fusion)** — gate/bias text–vision contributions when disagreement spikes; audit who influenced whom and why.
- **Cognitive & adaptive agents** — systems that self-correct against misalignment with interpretable state and policy switches.
- **Metacognitive mechanics** — self-monitoring, policy evaluation/evolution, and reflective control over when/how modules adapt.
- **Networked control** — orchestrate layers/heads/modules/sensors as nodes; propagate influence with decay/cooldowns for stable coordination.
- **Simulation & research** — explore rupture dynamics, policy A/B, and bounded evolution with reproducible logs.

---


## Install

```bash
pip install cognize
```

---

## Core primitives

| Symbol | Meaning               |
|:------:|-----------------------|
| `V`    | Belief / Projection   |
| `R`    | Reality signal        |
| `∆`    | Distortion (`R−V`)    |
| `Θ`    | Rupture threshold     |
| `E`    | Misalignment memory   |
| `⊙`    | Realignment operator  |

---

## Examples

### 1) Quick start (scalar)
```python
from cognize import EpistemicState
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay

state = EpistemicState(V0=0.5, threshold=0.35, realign_strength=0.3)
state.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)

for r in [0.1, 0.3, 0.7, 0.9]:
    state.receive(r)

print(state.explain_last())  # human-readable step summary
print(state.summary())       # compact state snapshot
```

### 2) Multimodal in one pass (vector)
```python
import numpy as np
from cognize import EpistemicState, Perception

def toy_text_encoder(s: str) -> np.ndarray:
    return np.array([len(s), s.count(" "), s.count("a"), 1.0], dtype=float)

P = Perception(text_encoder=toy_text_encoder)
state = EpistemicState(V0=np.zeros(4), perception=P)

state.receive({"text": "hello world"})
print(state.last())  # includes Δ, Θ, ruptured, etc.
```

### 3) Meta‑policy selection
```python
from cognize import EpistemicState, PolicyManager, PolicyMemory, ShadowRunner, SAFE_SPECS
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay

s = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.3)
s.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
s.policy_manager = PolicyManager(
    base_specs=SAFE_SPECS, memory=PolicyMemory(), shadow=ShadowRunner(),
    epsilon=0.15, promote_margin=1.03, cooldown_steps=30
)

for r in [0.2, 0.4, 0.5, 0.7, 0.6, 0.8]:
    s.receive(r)

print(s.summary())
```

### 4) CSV / JSON export & small stats
```python
from pathlib import Path
from statistics import mean
from cognize import EpistemicState
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay

s = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.3)
s.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)

for r in [0.1, 0.3, 0.9, 0.2, 0.8, 0.7]: s.receive(r)

out = Path("trace.csv"); s.export_csv(str(out))
print("ruptures:", s.summary()["ruptures"])
print("mean |Δ| (last 10):", mean(abs(h["∆"]) for h in s.history[-10:]))
```

### 5) Plain EpistemicGraph (coupling multiple states)
```python
from cognize import make_simple_state, EpistemicGraph

G = EpistemicGraph(damping=0.5, max_depth=2, max_step=1.0, rupture_only_propagation=True)
G.add("A", make_simple_state(0.0)); G.add("B", make_simple_state(0.0)); G.add("C", make_simple_state(0.0))

# A → B (pressure), B → C (delta)
G.link("A", "B", weight=0.8, mode="pressure", decay=0.9, cooldown=3)
G.link("B", "C", weight=0.5, mode="delta",    decay=0.9, cooldown=2)

# Step node A with evidence; influence cascades per edge modes
G.step("A", 1.2)
print(G.stats())
print("hot edges:", G.top_edges(by="applied_ema", k=5))
print("last cascade:", G.last_cascade(5))
```

### 6) Programmable graph: register a strategy and link by reference
```python
from typing import Dict, Any, Optional
import numpy as np
from cognize import EpistemicProgrammableGraph, register_strategy

# Minimal programmable pieces (use defaults for the rest)
def gate_fn(src_st, dst_st, ctx: Dict[str, Any]) -> bool:
    # fire only on rupture for 'pressure'/'policy'; always for 'delta'
    mode = ctx["edge"]["mode"]; rupt = bool(ctx["post_src"].get("ruptured", False))
    return (mode != "pressure" and mode != "policy") or rupt

def influence_fn(src_st, post_src: Dict[str, Any], ctx: Dict[str, Any]) -> float:
    delta, theta = float(post_src.get("∆", 0.0)), float(post_src.get("Θ", 0.0))
    return max(0.0, delta - theta)  # pressure

def target_fn(dst_st, edge_meta: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[slice]:
    # take middle half of a vector V if available
    if not isinstance(dst_st.V, np.ndarray): return None
    n = dst_st.V.shape[0]; i, j = n//4, 3*n//4
    return slice(i, j)

register_strategy("cooling@1.0.0", gate_fn=gate_fn, influence_fn=influence_fn, target_fn=target_fn)

G = EpistemicProgrammableGraph(damping=0.6, max_depth=2)
G.add("X"); G.add("Y")
# attach by reference; params are JSON-safe and persisted
G.link("X", "Y", mode="policy", weight=0.7, decay=0.9, cooldown=4,
       strategy_id="cooling@1.0.0", params={"bias_decay": 0.9})

# Drive X; programmable edge applies reversible Θ↑/k↓ bias on Y when X ruptures
G.step("X", 1.4)
print(G.last_cascade(3))

# Persist topology + strategy references (no code serialization)
G.save_graph("graph.json", include_strategies=True)

# Load later (rebinds strategies by ID from registry)
H = EpistemicProgrammableGraph()
H.add("X"); H.add("Y")
H.load_graph("graph.json", strict_strategies=False)
```

### 7) Influence preview (what would be applied?)
```python
from cognize import EpistemicGraph, make_simple_state

G = EpistemicGraph()
G.add("A", make_simple_state(0.0)); G.add("B", make_simple_state(0.0))
G.link("A", "B", weight=0.8, mode="pressure", decay=0.9, cooldown=1)

# Pretend A just ruptured with Δ=1.0, Θ=0.3 (no state mutation)
postA = {"∆": 1.0, "Θ": 0.3, "ruptured": True}
print("predicted magnitude:", G.predict_influence("A", "B", post=postA))
```

### 8) Suspend propagation (isolate learning vs. coupling)
```python
from cognize import EpistemicGraph, make_simple_state

G = EpistemicGraph()
for n in ("A","B"): G.add(n, make_simple_state(0.0))
G.link("A","B", weight=1.0, mode="pressure")

with G.suspend_propagation():
    # A will update itself, but won't influence B during this block
    G.step("A", 2.0)

# Propagation resumes here
G.step("A", 2.2)
```

### 9) Tiny NN control‑plane sketch (PyTorch, optional)
```python
# Pseudo-code: shows the observer → graph → nudge loop
import torch
from cognize import EpistemicProgrammableGraph

peg = EpistemicProgrammableGraph(max_depth=1, damping=0.5)
peg.add("L23"); peg.add("HEAD7")
peg.link("L23","HEAD7", mode="policy", weight=0.6, decay=0.9, cooldown=3)

def entropy(x):  # simple example metric
    p = torch.softmax(x.flatten(), dim=0); return -(p * (p+1e-9).log()).sum().item()

attn_logits_ref = {}  # cache last logits tensor per step (just illustrative)

def hook_L23(module, inp, out):
    peg.step("L23", {"norm": out.norm().item(), "ruptured": False})  # you decide the R fields

def hook_HEAD7(module, inp, out):
    attn_logits_ref["HEAD7"] = out  # capture a handle to nudge later

# Attach forward hooks on your model (where it makes sense)
# layer23.register_forward_hook(hook_L23)
# head7.register_forward_hook(hook_HEAD7)

# After forward:
# peg.step("HEAD7", {"entropy": entropy(attn_logits_ref["HEAD7"])})
# (peg runs propagation internally during step)
# Apply your bounded nudges here according to your edge strategies/logs.
```
---

## Citation

If you use **Cognize**, please cite the concept DOI (always resolves to the latest version):

```bibtex
@software{pulikanti_cognize,
  author    = {Pulikanti, Sashi Bharadwaj},
  title     = {Cognize: Programmable cognition for Python systems},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17042859},
  url       = {https://doi.org/10.5281/zenodo.17042859}
}
```

---

## License

Licensed under the **Apache License 2.0**.  
© 2025 Pulikanti Sashi Bharadwaj
