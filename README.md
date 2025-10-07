# AIND ANTs Transform Sidecar

![CI](https://github.com/AllenNeuralDynamics/aind-ants-transform-sidecar/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-ants-transform-sidecar)](https://pypi.org/project/aind-ants-transform-sidecar/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

Metadata for ANTs transform objects

## Overview

This package provides metadata "sidecars" for [ANTs (Advanced Normalization Tools)](http://stnava.github.io/ANTs/) transform objects. ANTs produces spatial transformations (affine matrices and displacement fields) for medical image registration, but the transform files alone don't document important context like:

- The spatial domains (bounding boxes, spacing, voxel dimensions) of the fixed and moving images
- The coordinate system and units being used
- How to correctly chain and apply multiple transformation steps

This package provides Pydantic models that bundle ANTs transform file paths with their spatial metadata, enabling:
- **Easy domain comparison**: Hash functions make it simple to verify domains match
- **Documentation**: Clear specification of coordinate frames (LPS), units (mm), and domain definitions
- **Interoperability**: Standardized JSON format for sharing transformation metadata
- **Convenience**: Easy conversion to ANTsPy function arguments for applying transforms

## Installation

If you choose to clone the repository, you can install the package by running the following command from the root directory of the repository:

```bash
pip install .
```

Otherwise, you can use pip:

```bash
pip install aind-ants-transform-sidecar
```

## Usage

```python
from aind_ants_transform_sidecar import (
    TransformSidecarV1,
    Domain,
    BBox,
    SynTriplet,
    load_package,
    dump_package,
)

# Define spatial domains for fixed and moving images
fixed_domain = Domain(
    spacing_LPS=(0.5, 0.5, 0.5),
    bbox=BBox(L=(-50.0, 50.0), P=(-50.0, 50.0), S=(-50.0, 50.0)),
    shape_canonical=(200, 200, 200),
)

moving_domain = Domain(
    spacing_LPS=(0.5, 0.5, 0.5),
    bbox=BBox(L=(-50.0, 50.0), P=(-50.0, 50.0), S=(-50.0, 50.0)),
    shape_canonical=(200, 200, 200),
)

# Create sidecar for ANTs SyN registration output
sidecar = TransformSidecarV1(
    fixed_domain=fixed_domain,
    moving_domain=moving_domain,
    transform=SynTriplet(
        affine="0GenericAffine.mat",
        warp="1Warp.nii.gz",
        inverse_warp="1InverseWarp.nii.gz",
    ),
)

# Serialize to JSON
json_str = dump_package(sidecar)

# Load from JSON
loaded = load_package(json_str)

# Get transformation chains
forward_chain = loaded.forward_chain()  # moving → fixed
inverse_chain = loaded.inverse_chain()  # fixed → moving

# Convert to ANTsPy arguments
transforms, whichtoinvert = forward_chain.antspy_apply_transforms_args()
```

## Development

To develop the code, run:
```bash
uv sync
```

## Development

Please test your changes using the full linting and testing suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual commands:
```bash
uv run --frozen ruff format          # Code formatting
uv run --frozen ruff check           # Linting
uv run --frozen mypy                 # Type checking
uv run --frozen interrogate -v       # Documentation coverage
uv run --frozen codespell --check-filenames  # Spell checking
uv run --frozen pytest --cov  # Tests with coverage
```


### Documentation
```bash
sphinx-build -b html docs/source/ docs/build/html
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
