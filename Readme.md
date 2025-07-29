
<p align="center">
  <img src="logo.png" alt="Cognize Logo" width="200"/>
</p>

# Cognize

**Give any Python system cognition.**

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Release](https://img.shields.io/badge/version-v0.1.0-informational)
![Status](https://img.shields.io/badge/status-beta-orange)

---

**Cognize** is a symbolic cognition layer for Python systems — from LLMs to agents to simulations.  
It enables programmable epistemic control by modeling belief (`V`), reality (`R`), misalignment (`∆`), memory (`E`), and rupture (`Θ`).

---

## Features

- Drift-aware cognition engine (`EpistemicState`)
- Programmable rupture thresholds and realignment logic
- Symbolic rupture and collapse modeling
- Supports high-dimensional reality inputs (e.g., embeddings)
- Export cognition logs (`.json`, `.csv`) for external audits
- Control layer for hallucination detection in LLMs or symbolic gating in agents
- Minimal, extensible, domain-agnostic

---

## Installation

```bash
pip install cognize
```

---

## Core Concepts

| Symbol | Meaning                |
|--------|------------------------|
| `V`    | Projection (belief)    |
| `R`    | Reality (signal)       |
| `∆`    | Distortion             |
| `Θ`    | Tolerance threshold    |
| `E`    | Misalignment memory    |
| `⊙`    | Stable                 |
| `⚠`    | Rupture                |
| `∅`    | No signal yet          |

---

## Example Usage

```python
from cognize import EpistemicState

e = EpistemicState(V0=0.0, threshold=0.4)

for R in [0.1, 0.3, 0.6, 0.8]:
    e.receive(R)
    print(e.symbol(), e.state())
```

**Expected Output:**
```
⊙ {'V': 0.03, 'E': 0.01, 'Θ': 0.4, ...}
⊙ {...}
⚠ {'V': 0.0, 'E': 0.0, 'Θ': 0.4, ...}
```

---

## License

Cognize is released under the [Apache 2.0 License](LICENSE).
