# Changelog
All notable changes to this project are documented here. This project follows the **Keep a Changelog** style and aims to respect **SemVer**.

## [Unreleased]

## [0.1.8] – 2025-09-06
### Added
- **`EpistemicProgrammableGraph`** with programmable edges *(gate → influence → magnitude → target(slice) → nudge → damp)*.
- **`PolicyManager`** + **`SAFE_SPECS`** with shadow evaluation and ε-greedy promotion.
- **`Perception`** adapter for multimodal vectors and `demo_text_encoder` helper.
- Convenience helpers: `make_simple_state`, `make_graph`; improved `__repr__`/`pulse()` for state/graph.

### Changed
- Packaging: Python **3.10+**, `numpy>=1.26`, `py.typed`, dynamic version single-sourced from code.
- Docs: README overhaul with NN control-plane sketch, examples, and concept-DOI citation.

### Fixed
- Minor robustness and logging improvements across export/summary paths.

## 0.1.3–0.1.7 (roll-up)
- Kernel hardening for `EpistemicState` (rupture, memory `E`, thresholds `Θ`, CSV/JSON export).
- Policy functions for `threshold`/`realign`/`collapse` and a small policy registry.
- Early graph utilities; examples and tests; packaging and CI polish.

## [0.1.2] – 2025-08-01
### Added
- `inject_policy(...)` on `EpistemicState` for dynamic override of rupture threshold, realignment, and collapse.
- Functional wrappers in `policies.py` for seamless injection (`threshold_adaptive_fn`, `realign_tanh_fn`, …).
- README/docs examples for programmable injection usage.

### Changed
- `epistemic.py` routes cognitive updates via injected policies; `receive()` accepts override logic.

### Fixed
- Improved symbolic output tracking for rupture vs. stable cognition cycles.

## [0.1.1] – 2025-07-31
### Added
- Base symbolic cognition engine (`EpistemicState`): drift tracking, misalignment memory, rupture detection.
- Export logs to `.json` and `.csv`.

## [0.1.0] – 2025-07-30
### Prototype
- Initial PyPI release (not tagged on GitHub): basic `EpistemicState`, core rupture logic, symbolic logging.

---

[Unreleased]: https://github.com/heraclitus0/cognize/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/heraclitus0/cognize/compare/v0.1.7...v0.1.8
[0.1.2]: https://github.com/heraclitus0/cognize/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/heraclitus0/cognize/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/heraclitus0/cognize/releases/tag/v0.1.0
