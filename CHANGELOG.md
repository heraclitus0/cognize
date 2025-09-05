# Changelog
All notable changes to this project are documented here.
This project follows the **Keep a Changelog** style and aims to respect **SemVer**.

## [Unreleased]

## [0.1.8] – 2025-09-06
### Added
- **EpistemicProgrammableGraph** with programmable edges (gate → influence → magnitude → target(slice) → nudge → damp).
- **PolicyManager** + **SAFE_SPECS** with shadow evaluation and ε-greedy promotion.
- **Perception** adapter for multimodal vectors and `demo_text_encoder` helper.
- Convenience helpers: `make_simple_state`, `make_graph`; improved `__repr__`/`pulse()` for state/graph.

### Changed
- Packaging: Python **3.10+**, `numpy>=1.26`, `py.typed`, dynamic version single-sourced from code.
- Docs: README overhaul with NN control-plane sketch, examples, and concept-DOI citation.

### Fixed
- Minor robustness and logging improvements across export/summary paths.

## 0.1.3–0.1.7 (roll-up)
- Kernel hardening for `EpistemicState` (rupture, memory `E`, thresholds `Θ`, CSV/JSON export).
- Policy functions for `threshold/realign/collapse` and a small policy registry.
- Early graph utilities; examples and tests; packaging and CI polish.

## [0.1.2] - 2025-08-01
### Added
- `inject_policy(...)` method to `EpistemicState` for dynamic override of rupture threshold, realignment logic, and collapse behavior.
- Self-compatible functional wrappers in `policies.py` for seamless injection.
- Support for symbolic threshold adaptation and bounded realignment via `threshold_adaptive_fn`, `realign_tanh_fn`, etc.
- README and documentation examples for programmable injection usage.

### Changed
- `epistemic.py` now dynamically routes cognitive updates based on injected policies.
- `receive()` method accepts override logic for drift, rupture, and projection realignment.

### Fixed
- Improved symbolic output tracking for rupture vs. stable cognition cycles.

---

## [0.1.1] - Initial Release
- Base symbolic cognition engine (`EpistemicState`)
- Supports drift tracking, misalignment memory, rupture detection.
- Export logs in `.json` and `.csv` format.

## [0.1.0] - 2025-07-30
### Prototype
- Initial PyPI release (not tagged on GitHub)
- Basic `EpistemicState` class with scalar drift tracking
- Core rupture logic and symbolic logging
