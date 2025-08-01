# Changelog

All notable changes to the Cognize project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.1] - 2025-08-01
### Added
- Modular injection support for rupture, realignment, and threshold logic
- `inject_policy()` method for programmable cognition behavior
- Built-in policies: `collapse_soft_decay`, `realign_tanh`, `threshold_adaptive` and more
- Vector signal support (multi-dimensional input)
- Export functions for `.json` and `.csv` cognition traces
- Complete unit test suite across all modules (100% coverage)
- Symbolic introspection (`summary()`, `drift_stats()`, `rupture_log()`)

### Changed
- Refined rupture handling: removed emoji symbols, improved semantic state tracking
- README redesigned for clarity and PyPI rendering
- Modularized epistemic logic into `epistemic.py` and `policies.py`

### Documentation
- Added `USER_GUIDE.md` with in-depth walkthrough and best practices
- Prepared structure for future DSL and projection interfaces

---

## [0.1.0] - 2025-07-28
### Prototype
- Initial PyPI release (not tagged on GitHub)
- Basic `EpistemicState` class with scalar drift tracking
- Core rupture logic and symbolic logging