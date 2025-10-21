# Changelog

All notable changes to Mussel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation improvements
  - Enhanced README.md with table of contents, quick start guide, and troubleshooting section
  - Expanded README-commands.md with detailed command descriptions and examples
  - Added CONTRIBUTING.md with development guidelines
  - Added CHANGELOG.md for tracking changes

### Changed
- Improved installation instructions with clearer system requirements
- Enhanced command documentation with detailed parameter descriptions
- Added visual examples and tips for each command

## [1.0.1] - Previous Release

### Added
- Support for multiple pathology foundation models:
  - ResNet-50
  - TransPath
  - Prov-GigaPath
  - Virchow
  - Virchow2
  - H-Optimus-0
  - OpenCLIP (QuiltNet)
  - GooglePath
  - Conch v1.5
- Zero-shot tissue-type annotation using CLIP
- Tile caching for efficient training data loading
- Export tiles as individual PNG files
- Filter features using trained classifiers
- Merge annotations from BMP files
- Linear probe benchmarking
- Model download and save functionality

### Changed
- Forked from CLAM with architectural improvements
- Updated tiling algorithm for better performance
- Made suitable for job submission systems (one script run per slide)
- Microns per pixel (mpp) parameter for consistent tiling

### Removed
- Original CLAM modeling components (focused on feature extraction and preprocessing)

## Notes

This project is a fork of [CLAM](https://github.com/mahmoodlab/CLAM) by the Mahmood Lab.
See the original repository for the history of changes prior to the fork.
