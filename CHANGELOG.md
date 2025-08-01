# Changelog

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
