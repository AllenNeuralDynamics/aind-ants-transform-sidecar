"""Tests for facade API (load_package and dump_package)."""

import json

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import (
    BBox,
    Domain,
    SynTriplet,
    TransformSidecarV1,
    dump_package,
    load_package,
)


@pytest.fixture
def minimal_sidecar_dict() -> dict:
    """Fixture providing minimal sidecar as dictionary."""
    return {
        "schema_version": "1.0",
        "frame": "LPS",
        "units": "mm",
        "transform": {
            "kind": "syn",
            "affine": "0GenericAffine.mat",
            "warp": "1Warp.nii.gz",
            "inverse_warp": "1InverseWarp.nii.gz",
        },
    }


@pytest.fixture
def full_sidecar_dict() -> dict:
    """Fixture providing full sidecar with domains as dictionary."""
    return {
        "schema_version": "1.0",
        "frame": "LPS",
        "units": "mm",
        "fixed_domain": {
            "definition": "voxel-center",
            "spacing": [1.0, 1.0, 1.0],
            "bbox": {"L": [-50.0, 50.0], "P": [-60.0, 40.0], "S": [-30.0, 70.0]},
            "shape_canonical": [100, 100, 100],
        },
        "moving_domain": {
            "definition": "voxel-center",
            "spacing": [1.0, 1.0, 1.0],
            "bbox": {"L": [-60.0, 60.0], "P": [-70.0, 50.0], "S": [-40.0, 80.0]},
            "shape_canonical": [120, 120, 120],
        },
        "transform": {
            "kind": "syn",
            "affine": "0GenericAffine.mat",
            "warp": "1Warp.nii.gz",
            "inverse_warp": "1InverseWarp.nii.gz",
        },
    }


class TestLoadPackage:
    """Test load_package function."""

    def test_load_from_json_string(self, minimal_sidecar_dict: dict) -> None:
        """Test loading package from JSON string."""
        json_str = json.dumps(minimal_sidecar_dict)
        model = load_package(json_str)

        assert isinstance(model, TransformSidecarV1)
        assert model.schema_version == "1.0"
        assert model.transform.affine == "0GenericAffine.mat"

    def test_load_from_dict(self, minimal_sidecar_dict: dict) -> None:
        """Test loading package from dictionary."""
        model = load_package(minimal_sidecar_dict)

        assert isinstance(model, TransformSidecarV1)
        assert model.schema_version == "1.0"
        assert model.transform.affine == "0GenericAffine.mat"

    def test_load_full_sidecar(self, full_sidecar_dict: dict) -> None:
        """Test loading full sidecar with domains."""
        model = load_package(full_sidecar_dict)

        assert isinstance(model, TransformSidecarV1)
        assert model.fixed_domain is not None
        assert model.moving_domain is not None
        assert model.fixed_domain.spacing == (1.0, 1.0, 1.0)
        assert model.moving_domain.spacing == (1.0, 1.0, 1.0)

    def test_load_rejects_missing_schema_version(self) -> None:
        """Test that load_package rejects data missing schema_version."""
        data = {
            "frame": "LPS",
            "units": "mm",
            "transform": {
                "kind": "syn",
                "affine": "0GenericAffine.mat",
                "warp": "1Warp.nii.gz",
                "inverse_warp": "1InverseWarp.nii.gz",
            },
        }
        with pytest.raises(ValueError, match="Missing 'schema_version'"):
            load_package(data)

    def test_load_rejects_unsupported_version(self) -> None:
        """Test that load_package rejects unsupported schema versions."""
        data = {
            "schema_version": "2.0",
            "frame": "LPS",
            "units": "mm",
            "transform": {
                "kind": "syn",
                "affine": "0GenericAffine.mat",
                "warp": "1Warp.nii.gz",
                "inverse_warp": "1InverseWarp.nii.gz",
            },
        }
        with pytest.raises(ValueError, match="Unsupported schema_version: 2.0"):
            load_package(data)

    def test_load_accepts_version_1_variants(self, minimal_sidecar_dict: dict) -> None:
        """Test that load_package accepts version '1.x' variants."""
        # Test 1.0
        model = load_package(minimal_sidecar_dict)
        assert model.schema_version == "1.0"

        # Currently only "1.0" is accepted due to Literal constraint
        # Future versions like "1.1" would need schema migration logic
        # Test that "1.0" string matching works
        data_10 = minimal_sidecar_dict.copy()
        data_10["schema_version"] = "1.0"
        model = load_package(data_10)
        assert isinstance(model, TransformSidecarV1)

    def test_load_validates_transform_data(self) -> None:
        """Test that load_package validates transform data."""
        data = {
            "schema_version": "1.0",
            "frame": "LPS",
            "units": "mm",
            "transform": {
                "kind": "syn",
                "affine": "",  # Invalid: empty affine
                "warp": "1Warp.nii.gz",
                "inverse_warp": "1InverseWarp.nii.gz",
            },
        }
        with pytest.raises(ValidationError):
            load_package(data)

    def test_load_validates_domain_data(self) -> None:
        """Test that load_package validates domain data."""
        data = {
            "schema_version": "1.0",
            "frame": "LPS",
            "units": "mm",
            "fixed_domain": {
                "spacing": [0.0, 1.0, 1.0],  # Invalid: zero spacing
                "bbox": {"L": [-50.0, 50.0], "P": [-60.0, 40.0], "S": [-30.0, 70.0]},
            },
            "transform": {
                "kind": "syn",
                "affine": "0GenericAffine.mat",
                "warp": "1Warp.nii.gz",
                "inverse_warp": "1InverseWarp.nii.gz",
            },
        }
        with pytest.raises(ValidationError):
            load_package(data)

    def test_load_with_spatial_signature(self) -> None:
        """Test loading sidecar with spatial signatures."""
        # First create a domain to get the signature
        bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=bbox)

        data = {
            "schema_version": "1.0",
            "frame": "LPS",
            "units": "mm",
            "fixed_domain": {
                "spacing": [1.0, 1.0, 1.0],
                "bbox": {"L": [-50.0, 50.0], "P": [-60.0, 40.0], "S": [-30.0, 70.0]},
                "spatial_signature": {
                    "method": domain.spatial_signature.method,
                    "blake2b": domain.spatial_signature.blake2b,
                },
            },
            "moving_domain": {
                "spacing": [1.0, 1.0, 1.0],
                "bbox": {"L": [-50.0, 50.0], "P": [-60.0, 40.0], "S": [-30.0, 70.0]},
            },
            "transform": {
                "kind": "syn",
                "affine": "0GenericAffine.mat",
                "warp": "1Warp.nii.gz",
                "inverse_warp": "1InverseWarp.nii.gz",
            },
        }
        model = load_package(data)
        assert model.fixed_domain is not None
        assert model.fixed_domain.spatial_signature == domain.spatial_signature


class TestDumpPackage:
    """Test dump_package function."""

    def test_dump_minimal_sidecar(self) -> None:
        """Test dumping minimal sidecar."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        sidecar = TransformSidecarV1(transform=triplet)
        json_str = dump_package(sidecar)

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["schema_version"] == "1.0"
        assert data["transform"]["kind"] == "syn"

    def test_dump_excludes_none_fields(self) -> None:
        """Test that dump_package excludes None fields."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        sidecar = TransformSidecarV1(transform=triplet)
        json_str = dump_package(sidecar)

        data = json.loads(json_str)
        assert "fixed_domain" not in data
        assert "moving_domain" not in data

    def test_dump_full_sidecar(self) -> None:
        """Test dumping full sidecar with domains."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        fixed_bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        moving_bbox = BBox(L=(-60.0, 60.0), P=(-70.0, 50.0), S=(-40.0, 80.0))
        fixed_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)
        moving_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=moving_bbox)

        sidecar = TransformSidecarV1(transform=triplet, fixed_domain=fixed_domain, moving_domain=moving_domain)
        json_str = dump_package(sidecar)

        data = json.loads(json_str)
        assert "fixed_domain" in data
        assert "moving_domain" in data
        assert data["fixed_domain"]["spacing"] == [1.0, 1.0, 1.0]

    def test_dump_includes_spatial_signatures(self) -> None:
        """Test that dump_package includes spatial signatures."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        fixed_bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        fixed_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)
        moving_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)

        sidecar = TransformSidecarV1(transform=triplet, fixed_domain=fixed_domain, moving_domain=moving_domain)
        json_str = dump_package(sidecar)

        data = json.loads(json_str)
        assert "spatial_signature" in data["fixed_domain"]
        assert "spatial_signature" in data["moving_domain"]
        assert "blake2b" in data["fixed_domain"]["spatial_signature"]


class TestRoundtrip:
    """Test load/dump roundtrip."""

    def test_roundtrip_minimal_sidecar(self, minimal_sidecar_dict: dict) -> None:
        """Test roundtrip of minimal sidecar."""
        model1 = load_package(minimal_sidecar_dict)
        json_str = dump_package(model1)
        model2 = load_package(json_str)

        assert model2.schema_version == model1.schema_version
        assert model2.frame == model1.frame
        assert model2.units == model1.units
        assert model2.transform.affine == model1.transform.affine
        assert model2.transform.warp == model1.transform.warp
        assert model2.transform.inverse_warp == model1.transform.inverse_warp

    def test_roundtrip_full_sidecar(self, full_sidecar_dict: dict) -> None:
        """Test roundtrip of full sidecar with domains."""
        model1 = load_package(full_sidecar_dict)
        json_str = dump_package(model1)
        model2 = load_package(json_str)

        assert model2.fixed_domain is not None
        assert model2.moving_domain is not None
        assert model1.fixed_domain is not None
        assert model1.moving_domain is not None

        assert model2.fixed_domain.spacing == model1.fixed_domain.spacing
        assert model2.moving_domain.spacing == model1.moving_domain.spacing
        assert model2.fixed_domain.spatial_signature == model1.fixed_domain.spatial_signature
        assert model2.moving_domain.spatial_signature == model1.moving_domain.spatial_signature

    def test_roundtrip_preserves_chains(self) -> None:
        """Test that roundtrip preserves chain generation."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        sidecar1 = TransformSidecarV1(transform=triplet)

        json_str = dump_package(sidecar1)
        sidecar2 = load_package(json_str)

        # Check forward chain
        chain1 = sidecar1.forward_chain()
        chain2 = sidecar2.forward_chain()
        assert len(chain1.steps) == len(chain2.steps)
        assert chain1.steps[0].file == chain2.steps[0].file
        assert chain1.steps[1].file == chain2.steps[1].file

        # Check inverse chain
        inv_chain1 = sidecar1.inverse_chain()
        inv_chain2 = sidecar2.inverse_chain()
        assert len(inv_chain1.steps) == len(inv_chain2.steps)
        assert inv_chain1.steps[0].file == inv_chain2.steps[0].file
        assert inv_chain1.steps[1].file == inv_chain2.steps[1].file

    def test_load_dump_is_idempotent(self, minimal_sidecar_dict: dict) -> None:
        """Test that multiple load/dump cycles produce same result."""
        model1 = load_package(minimal_sidecar_dict)
        json1 = dump_package(model1)

        model2 = load_package(json1)
        json2 = dump_package(model2)

        # Parse both JSON strings to compare structure (ignore whitespace differences)
        data1 = json.loads(json1)
        data2 = json.loads(json2)
        assert data1 == data2
