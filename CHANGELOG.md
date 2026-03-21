# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-03-21

### Added

- Binder integration for running notebooks in the browser
- Google Colab support with auto-download setup cell for each notebook
- `.gitignore` for Python and Jupyter artifacts
- `requirements.txt` with project dependencies
- `CONTRIBUTING.md` with contribution guidelines
- `SECURITY.md` with vulnerability reporting policy
- `SUPPORT.md` with support channels
- `CHANGELOG.md` to track project changes
- `CITATION.cff` for academic citation
- GitHub Issue templates (bug report, feature request)
- GitHub Pull Request template

### Changed

- Reorganized project into `notebooks/`, `data/`, and `images/` directories
- Renamed notebooks with descriptive, numbered names (e.g., `Part1_1.ipynb` -> `01-perceptron-basics.ipynb`)
- Renamed data files to lowercase with typo fix (`heat_influx_noth_south.csv` -> `heat_influx_north_south.csv`)
- Renamed image files with cleaner, lowercase names
- Rewrote `README.md` with professional structure, accurate file paths, and online access badges
- Updated GitHub repository description and topics
- Updated all internal CSV path references in notebooks

### Removed

- Redundant backup notebook copies (`*copy.ipynb`)
- Placeholder images from README
- Unused social share links and excessive decorative elements

## [0.1.0] - 2024-05-21

### Added

- Initial implementation of perceptron model for fish species classification
- Linear neuron models for heat influx prediction (single-input and multi-input)
- Gradient descent optimization with learning rate tuning
- 3D visualization of prediction surfaces
- Fish classification dataset (`Fish_data.csv`)
- Heat influx dataset (`heat_influx_noth_south.csv`)
- MIT License
- Code of Conduct (Contributor Covenant v2.0)
