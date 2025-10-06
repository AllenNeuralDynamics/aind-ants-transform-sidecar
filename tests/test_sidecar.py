"""Tests for TransformSidecarV1 model."""

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import BBox, Domain, SynTriplet, TransformSidecarV1


@pytest.fixture
def valid_bbox() -> BBox:
    """Fixture providing a valid BBox."""
    return BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))


@pytest.fixture
def valid_domain(valid_bbox: BBox) -> Domain:
    """Fixture providing a valid Domain."""
    return Domain(spacing=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))


@pytest.fixture
def valid_syn_triplet() -> SynTriplet:
    """Fixture providing a valid SynTriplet."""
    return SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")


class TestTransformSidecarV1Validation:
    """Test TransformSidecarV1 model validation."""

    def test_minimal_sidecar_creation(self, valid_syn_triplet: SynTriplet) -> None:
        """Test creating a minimal sidecar without domains."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        assert sidecar.schema_version == "1.0"
        assert sidecar.frame == "LPS"
        assert sidecar.units == "mm"
        assert sidecar.fixed_domain is None
        assert sidecar.moving_domain is None
        assert sidecar.transform == valid_syn_triplet

    def test_full_sidecar_creation(self, valid_syn_triplet: SynTriplet, valid_domain: Domain) -> None:
        """Test creating a full sidecar with domains."""
        # Create two different domains for fixed and moving
        fixed_bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        moving_bbox = BBox(L=(-60.0, 60.0), P=(-70.0, 50.0), S=(-40.0, 80.0))
        fixed_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)
        moving_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=moving_bbox)

        sidecar = TransformSidecarV1(
            transform=valid_syn_triplet, fixed_domain=fixed_domain, moving_domain=moving_domain
        )
        assert sidecar.fixed_domain == fixed_domain
        assert sidecar.moving_domain == moving_domain

    def test_sidecar_rejects_only_fixed_domain(self, valid_syn_triplet: SynTriplet, valid_domain: Domain) -> None:
        """Test that sidecar rejects having only fixed_domain without moving_domain."""
        with pytest.raises(ValidationError, match="Provide both fixed_domain and moving_domain or neither"):
            TransformSidecarV1(transform=valid_syn_triplet, fixed_domain=valid_domain)

    def test_sidecar_rejects_only_moving_domain(self, valid_syn_triplet: SynTriplet, valid_domain: Domain) -> None:
        """Test that sidecar rejects having only moving_domain without fixed_domain."""
        with pytest.raises(ValidationError, match="Provide both fixed_domain and moving_domain or neither"):
            TransformSidecarV1(transform=valid_syn_triplet, moving_domain=valid_domain)

    def test_schema_version_is_locked(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that schema_version is locked to '1.0'."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        assert sidecar.schema_version == "1.0"

    def test_frame_is_locked_to_lps(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that frame is locked to 'LPS'."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        assert sidecar.frame == "LPS"

    def test_units_is_locked_to_mm(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that units is locked to 'mm'."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        assert sidecar.units == "mm"


class TestTransformSidecarV1Chains:
    """Test TransformSidecarV1 chain delegation."""

    def test_forward_chain_delegates_to_transform(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that forward_chain delegates to the transform."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        chain = sidecar.forward_chain()

        # Should match the transform's forward_chain
        expected_chain = valid_syn_triplet.forward_chain()
        assert len(chain.steps) == len(expected_chain.steps)
        assert chain.steps[0].file == expected_chain.steps[0].file
        assert chain.steps[1].file == expected_chain.steps[1].file

    def test_inverse_chain_delegates_to_transform(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that inverse_chain delegates to the transform."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        chain = sidecar.inverse_chain()

        # Should match the transform's inverse_chain
        expected_chain = valid_syn_triplet.inverse_chain()
        assert len(chain.steps) == len(expected_chain.steps)
        assert chain.steps[0].file == expected_chain.steps[0].file
        assert chain.steps[1].file == expected_chain.steps[1].file

    def test_forward_chain_structure(self, valid_syn_triplet: SynTriplet) -> None:
        """Test forward chain has expected structure."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        chain = sidecar.forward_chain()

        assert len(chain.steps) == 2
        assert chain.steps[0].kind == "displacement_field"
        assert chain.steps[0].role == "forward"
        assert chain.steps[1].kind == "affine"
        assert chain.steps[1].invert is False

    def test_inverse_chain_structure(self, valid_syn_triplet: SynTriplet) -> None:
        """Test inverse chain has expected structure."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        chain = sidecar.inverse_chain()

        assert len(chain.steps) == 2
        assert chain.steps[0].kind == "affine"
        assert chain.steps[0].invert is True
        assert chain.steps[1].kind == "displacement_field"
        assert chain.steps[1].role == "inverse"


class TestTransformSidecarV1Serialization:
    """Test TransformSidecarV1 serialization."""

    def test_minimal_sidecar_serialization(self, valid_syn_triplet: SynTriplet) -> None:
        """Test serialization of minimal sidecar."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        json_data = sidecar.model_dump()

        assert json_data["schema_version"] == "1.0"
        assert json_data["frame"] == "LPS"
        assert json_data["units"] == "mm"
        assert json_data["transform"]["kind"] == "syn"

    def test_minimal_sidecar_excludes_none(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that model_dump with exclude_none removes None fields."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        json_data = sidecar.model_dump(exclude_none=True)

        assert "fixed_domain" not in json_data
        assert "moving_domain" not in json_data

    def test_full_sidecar_serialization(self, valid_syn_triplet: SynTriplet) -> None:
        """Test serialization of full sidecar with domains."""
        fixed_bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        moving_bbox = BBox(L=(-60.0, 60.0), P=(-70.0, 50.0), S=(-40.0, 80.0))
        fixed_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)
        moving_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=moving_bbox)

        sidecar = TransformSidecarV1(
            transform=valid_syn_triplet, fixed_domain=fixed_domain, moving_domain=moving_domain
        )
        json_data = sidecar.model_dump()

        assert "fixed_domain" in json_data
        assert "moving_domain" in json_data
        # Pydantic serializes tuples as tuples, not lists
        assert json_data["fixed_domain"]["spacing"] == (1.0, 1.0, 1.0)
        assert json_data["moving_domain"]["spacing"] == (1.0, 1.0, 1.0)

    def test_sidecar_roundtrip_minimal(self, valid_syn_triplet: SynTriplet) -> None:
        """Test roundtrip serialization of minimal sidecar."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        json_data = sidecar.model_dump()
        sidecar2 = TransformSidecarV1.model_validate(json_data)

        assert sidecar2.schema_version == sidecar.schema_version
        assert sidecar2.frame == sidecar.frame
        assert sidecar2.units == sidecar.units
        assert sidecar2.transform.affine == sidecar.transform.affine
        assert sidecar2.transform.warp == sidecar.transform.warp
        assert sidecar2.transform.inverse_warp == sidecar.transform.inverse_warp

    def test_sidecar_roundtrip_full(self, valid_syn_triplet: SynTriplet) -> None:
        """Test roundtrip serialization of full sidecar with domains."""
        fixed_bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        moving_bbox = BBox(L=(-60.0, 60.0), P=(-70.0, 50.0), S=(-40.0, 80.0))
        fixed_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=fixed_bbox)
        moving_domain = Domain(spacing=(1.0, 1.0, 1.0), bbox=moving_bbox)

        sidecar = TransformSidecarV1(
            transform=valid_syn_triplet, fixed_domain=fixed_domain, moving_domain=moving_domain
        )
        json_data = sidecar.model_dump()
        sidecar2 = TransformSidecarV1.model_validate(json_data)

        assert sidecar2.fixed_domain is not None
        assert sidecar2.moving_domain is not None
        assert sidecar2.fixed_domain.spacing == fixed_domain.spacing
        assert sidecar2.moving_domain.spacing == moving_domain.spacing
        assert sidecar2.fixed_domain.spatial_signature == fixed_domain.spatial_signature
        assert sidecar2.moving_domain.spatial_signature == moving_domain.spatial_signature

    def test_sidecar_json_serialization(self, valid_syn_triplet: SynTriplet) -> None:
        """Test JSON string serialization."""
        sidecar = TransformSidecarV1(transform=valid_syn_triplet)
        json_str = sidecar.model_dump_json(exclude_none=True)

        assert isinstance(json_str, str)
        assert '"schema_version":"1.0"' in json_str or '"schema_version": "1.0"' in json_str
        assert '"frame":"LPS"' in json_str or '"frame": "LPS"' in json_str
        assert '"kind":"syn"' in json_str or '"kind": "syn"' in json_str
