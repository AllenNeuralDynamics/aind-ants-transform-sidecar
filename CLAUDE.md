# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for creating metadata for ANTs (Advanced Normalization Tools) transform objects. It provides Pydantic models for representing spatial transformations (particularly ANTs SyN registration outputs) with layout-invariant domain definitions and cryptographic signatures for reproducibility.

**Package:** `aind-ants-transform-sidecar`
**Import name:** `aind_ants_transform_sidecar`
**Python version:** 3.10+ (target: 3.13)
**Primary dependency:** Pydantic 2.x

### What is ANTs?
ANTs (Advanced Normalization Tools) is a medical image registration toolkit. This package provides metadata schemas for ANTs transformation outputs, particularly SyN (Symmetric Normalization) registration results which typically produce:
- An affine transform (`.mat` file)
- A forward warp field (`Warp.nii.gz`)
- An inverse warp field (`InverseWarp.nii.gz`)

## Development Setup

Install dependencies and set up the environment:
```bash
uv sync
```

The project uses `uv` for dependency management with dependency groups defined in `pyproject.toml`.

## Common Development Commands

### Running the full linting and testing suite
```bash
./scripts/run_linters_and_checks.sh -c
```

This script runs all linters and checks. Without `-c`, it only runs `ruff format`.

### Individual commands (preferred for specific tasks)
```bash
# Code formatting (auto-fix)
uv run --frozen ruff format

# Linting
uv run --frozen ruff check

# Type checking
uv run --frozen mypy

# Documentation coverage check
uv run --frozen interrogate -v

# Spell checking
uv run --frozen codespell --check-filenames

# Run all tests with coverage
uv run --frozen pytest --cov

# Run a single test file
uv run --frozen pytest tests/test_example.py

# Run a specific test function
uv run --frozen pytest tests/test_example.py::test_version
```

### Documentation
Build the documentation:
```bash
sphinx-build -b html docs/source/ docs/build/html
```

## Code Standards

### Type Hints
- Type hints are **required** for all source code in `src/`
- Type checking is enforced with `mypy` with strict settings
- Tests in `tests/` are exempt from type annotation requirements (per-file ignore in ruff config)

### Code Style
- **Line length:** 120 characters
- **Style guide:** Enforced via ruff with extended rule sets (Q, RUF100, C90, I, F, E, W, UP, ANN, PYI)
- **Docstring convention:** NumPy style
- **Documentation coverage:** Minimum 30% (interrogate threshold)

### Testing
- Tests use pytest with coverage reporting
- Coverage threshold is currently set to 0% but should be increased as the project matures
- Test files must match `test_*.py` pattern
- Test classes must match `Test*` pattern
- Test functions must match `test_*` pattern

## Architecture

### Core Data Models (src/aind_ants_transform_sidecar/sidecar.py)

The module is organized into several layers:

#### 1. Spatial Domain Models
- **`BBox`**: Bounding box in LPS coordinates (Left-Posterior-Superior medical imaging convention)
  - Validates that min ≤ max for each axis (L, P, S)
  - Ensures all values are finite

- **`ContentSignature`**: Cryptographic hash for spatial domains
  - Uses BLAKE2b hashing for reproducibility
  - Method: quantize floats to 9 decimal places → pack as big-endian int64 → hash

- **`Domain`**: Layout-invariant spatial domain specification
  - `definition`: Always "voxel-center" (defines coordinate interpretation)
  - `spacing`: Voxel spacing in mm (3D tuple)
  - `bbox`: Bounding box in LPS coordinates
  - `shape_canonical`: Optional voxel dimensions (3D tuple of ints)
  - `spatial_signature`: Auto-computed BLAKE2b hash of geometry (auto-populated/validated)
  - **Key behavior**: Spatial signature is automatically computed on construction and verified if provided

#### 2. Transform Chain Models
- **`StepAffine`**: Affine transformation step
  - `file`: Path to `.mat` file
  - `invert`: Whether to invert the transform

- **`StepField`**: Displacement field transformation step
  - `file`: Path to `.nii.gz` warp file
  - `role`: "forward" or "inverse"
  - `grid`: Always "fixed" (ANTs SyN constraint)

- **`Chain`**: Ordered sequence of transformation steps
  - `order`: Always "top_to_bottom"
  - `steps`: List of StepAffine or StepField (discriminated union)
  - `antspy_apply_transforms_args()`: Converts chain to ANTsPy function arguments

#### 3. Operation Models
- **`SynTriplet`**: ANTs SyN registration output (affine + forward/inverse warps)
  - `affine`: Path to affine transform file
  - `warp`: Path to forward warp field
  - `inverse_warp`: Path to inverse warp field
  - `forward_chain()`: Returns Chain for moving→fixed transformation
  - `inverse_chain()`: Returns Chain for fixed→moving transformation
  - **Operation union**: Currently only SynTriplet, but designed for extension

#### 4. Top-Level Sidecar
- **`TransformSidecarV1`**: Main schema (v1.0)
  - `schema_version`: "1.0"
  - `frame`: "LPS" (coordinate system)
  - `units`: "mm"
  - `fixed_domain`: Optional Domain for fixed image
  - `moving_domain`: Optional Domain for moving image (must be paired with fixed_domain)
  - `transform`: Operation (currently SynTriplet)
  - Provides `forward_chain()` and `inverse_chain()` accessors

#### 5. Facade API
- **`load_package(src)`**: Load from JSON string or dict
  - Validates schema_version
  - Returns InternalModel (currently TransformSidecarV1)
  - Designed for future schema version migration

- **`dump_package(model)`**: Serialize to JSON
  - Excludes None fields
  - Uses Pydantic aliases if defined

### Design Patterns

1. **Discriminated Unions**: `Step` and `Operation` use Pydantic's discriminator pattern (via `kind` field)
2. **Auto-validation**: Pydantic validators enforce constraints (monotonic bbox, positive spacing, paired domains, etc.)
3. **Crypto Hashing**: Spatial domains use quantized big-endian representation for cross-platform reproducible hashing
4. **Forward Compatibility**: Version-based loading supports future schema migrations
5. **Coordinate Convention**: All spatial data in LPS (medical imaging standard)

## Project Structure

```
src/aind_ants_transform_sidecar/
├── __init__.py              # Package version defined here
├── sidecar.py               # Core Pydantic models and facade API (237 lines)
└── scripts/                 # CLI scripts (configured in pyproject.toml)
    └── __init__.py

tests/
└── test_example.py          # Basic version test (needs expansion)

docs/                        # Sphinx documentation
scripts/                     # Development scripts
```

## Version Management

- Version is defined in `src/aind_ants_transform_sidecar/__init__.py` as `__version__`
- Project uses commitizen for version bumping
- Semantic versioning with major version zero enabled
- Currently at version 0.1.0

## CI/CD

The project uses GitHub Actions workflows from `AllenNeuralDynamics/galen-uv-workflows`:
- CI checks on pull requests
- Documentation building
- Version bumping (semantic-release with Angular commit style)
- Publishing (configured but disabled: `publish_to_pypi: false`)

## Important Notes

- This project was generated from a Copier template. Do not manually edit `.copier-answers.yml`
- The project is managed with `uv`, so always use `uv run --frozen` for commands to avoid accidental dependency updates
- When adding CLI scripts, register them in `[project.scripts]` section of `pyproject.toml`
