# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com), and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- **2024-11-19**: Added an option to specify the calibration data save path. The default path is the system temporary directory (`/tmp/` for Unix-like systems, `~/AppData/Local/Temp` for Windows).
- **2024-11-18**: Added support for the `mono` subcommand to enable monocular camera calibration. The tool now supports both mono and stereo calibration modes.

### Fixed
- **2024-11-19**: Fixed a bug where clicking the OpenCV window close button did not terminate the program. This resolves a legacy issue related to ROS calibration.

### Changed
- Introduced separate subcommands for calibration, providing more flexibility. Previously, only stereo calibration was supported.
