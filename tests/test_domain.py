"""Tests for Domain model and spatial hashing."""

import math

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import BBox, ContentSignature, Domain


@pytest.fixture
def valid_bbox() -> BBox:
    """Fixture providing a valid BBox."""
    return BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))


@pytest.fixture
def valid_domain(valid_bbox: BBox) -> Domain:
    """Fixture providing a valid Domain."""
    return Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))


class TestDomainValidation:
    """Test Domain model validation."""

    def test_valid_domain_creation(self, valid_bbox: BBox) -> None:
        """Test creating a valid Domain."""
        domain = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))
        assert domain.spacing_LPS == (1.0, 1.0, 1.0)
        assert domain.bbox == valid_bbox
        assert domain.shape_canonical == (100, 100, 100)
        assert domain.definition == "voxel-center"
        assert domain.spatial_signature is not None

    def test_domain_without_shape_canonical(self, valid_bbox: BBox) -> None:
        """Test Domain creation without shape_canonical (optional field)."""
        domain = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        assert domain.spacing_LPS == (1.0, 1.0, 1.0)
        assert domain.bbox == valid_bbox
        assert domain.shape_canonical is None
        assert domain.spatial_signature is not None

    def test_domain_rejects_zero_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects zero spacing_LPS."""
        with pytest.raises(ValidationError, match="spacing_LPS must be positive and finite"):
            Domain(spacing_LPS=(0.0, 1.0, 1.0), bbox=valid_bbox)

    def test_domain_rejects_negative_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects negative spacing_LPS."""
        with pytest.raises(ValidationError, match="spacing_LPS must be positive and finite"):
            Domain(spacing_LPS=(1.0, -1.0, 1.0), bbox=valid_bbox)

    def test_domain_rejects_infinite_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects infinite spacing_LPS."""
        with pytest.raises(ValidationError, match="spacing_LPS must be positive and finite"):
            Domain(spacing_LPS=(math.inf, 1.0, 1.0), bbox=valid_bbox)

    def test_domain_rejects_nan_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects NaN spacing_LPS."""
        with pytest.raises(ValidationError, match="spacing_LPS must be positive and finite"):
            Domain(spacing_LPS=(math.nan, 1.0, 1.0), bbox=valid_bbox)

    def test_domain_rejects_negative_shape_canonical(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects negative shape_canonical values."""
        with pytest.raises(ValidationError, match="shape_canonical must be positive"):
            Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(-100, 100, 100))

    def test_domain_rejects_zero_shape_canonical(self, valid_bbox: BBox) -> None:
        """Test that Domain rejects zero shape_canonical values."""
        with pytest.raises(ValidationError, match="shape_canonical must be positive"):
            Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 0, 100))

    def test_domain_with_anisotropic_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test Domain with different spacing_LPS per axis."""
        domain = Domain(spacing_LPS=(0.5, 1.0, 2.0), bbox=valid_bbox)
        assert domain.spacing_LPS == (0.5, 1.0, 2.0)

    def test_domain_with_very_small_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test Domain with very small spacing_LPS values."""
        domain = Domain(spacing_LPS=(0.001, 0.001, 0.001), bbox=valid_bbox)
        assert domain.spacing_LPS == (0.001, 0.001, 0.001)

    def test_domain_with_large_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test Domain with large spacing_LPS values."""
        domain = Domain(spacing_LPS=(100.0, 100.0, 100.0), bbox=valid_bbox)
        assert domain.spacing_LPS == (100.0, 100.0, 100.0)


class TestSpatialHashing:
    """Test spatial signature hashing behavior."""

    def test_spatial_signature_auto_populated(self, valid_bbox: BBox) -> None:
        """Test that spatial_signature is auto-populated when None."""
        domain = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        assert domain.spatial_signature is not None
        assert domain.spatial_signature.method.startswith("LPS_bbox_spacing_shape")
        assert domain.spatial_signature.blake2b.startswith("b2:")

    def test_spatial_signature_reproducible(self, valid_bbox: BBox) -> None:
        """Test that spatial signature is reproducible for same inputs."""
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))
        domain2 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))
        assert domain1.spatial_signature == domain2.spatial_signature

    def test_spatial_signature_changes_with_spacing_LPS(self, valid_bbox: BBox) -> None:
        """Test that spatial signature changes when spacing_LPS changes."""
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        domain2 = Domain(spacing_LPS=(1.1, 1.0, 1.0), bbox=valid_bbox)
        assert domain1.spatial_signature != domain2.spatial_signature

    def test_spatial_signature_changes_with_bbox(self) -> None:
        """Test that spatial signature changes when bbox changes."""
        bbox1 = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        bbox2 = BBox(L=(-50.0, 51.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=bbox1)
        domain2 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=bbox2)
        assert domain1.spatial_signature != domain2.spatial_signature

    def test_spatial_signature_changes_with_shape(self, valid_bbox: BBox) -> None:
        """Test that spatial signature changes when shape changes."""
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))
        domain2 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(101, 100, 100))
        assert domain1.spatial_signature != domain2.spatial_signature

    def test_spatial_signature_differs_with_without_shape(self, valid_bbox: BBox) -> None:
        """Test that spatial signature differs when shape is present vs absent."""
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=(100, 100, 100))
        domain2 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, shape_canonical=None)
        assert domain1.spatial_signature != domain2.spatial_signature

    def test_spatial_signature_validation_accepts_correct(self, valid_bbox: BBox) -> None:
        """Test that providing correct spatial_signature is accepted."""
        # First create domain to get the signature
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        sig = domain1.spatial_signature

        # Now create with the same signature explicitly
        domain2 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, spatial_signature=sig)
        assert domain2.spatial_signature == sig

    def test_spatial_signature_validation_rejects_mismatch(self, valid_bbox: BBox) -> None:
        """Test that providing incorrect spatial_signature is rejected."""
        wrong_sig = ContentSignature(method="wrong_method", blake2b="b2:0123456789abcdef0123456789abcdef")
        with pytest.raises(ValidationError, match="spatial_signature mismatch"):
            Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox, spatial_signature=wrong_sig)

    def test_spatial_hash_sensitivity_to_small_changes(self, valid_bbox: BBox) -> None:
        """Test that hash is sensitive to small floating point changes."""
        # Change spacing_LPS by a tiny amount (well within hash precision of 9 decimal places)
        domain1 = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        domain2 = Domain(spacing_LPS=(1.0000000001, 1.0, 1.0), bbox=valid_bbox)
        # Should be same (within 9 decimal places)
        assert domain1.spatial_signature == domain2.spatial_signature

        # Change spacing_LPS beyond hash precision (10^-9)
        domain3 = Domain(spacing_LPS=(1.00000001, 1.0, 1.0), bbox=valid_bbox)
        assert domain1.spatial_signature != domain3.spatial_signature

    def test_spatial_signature_method_includes_config(self, valid_bbox: BBox) -> None:
        """Test that spatial signature method string reflects configuration."""
        domain = Domain(spacing_LPS=(1.0, 1.0, 1.0), bbox=valid_bbox)
        assert "intQ9" in domain.spatial_signature.method  # 9 decimal digits
        assert "BE" in domain.spatial_signature.method  # big-endian

    def test_spatial_hash_with_extreme_values(self) -> None:
        """Test spatial hashing with extreme but valid coordinate values."""
        bbox = BBox(L=(-1e6, 1e6), P=(-1e6, 1e6), S=(-1e6, 1e6))
        domain = Domain(spacing_LPS=(100.0, 100.0, 100.0), bbox=bbox)
        assert domain.spatial_signature is not None
        assert domain.spatial_signature.blake2b.startswith("b2:")

    def test_roundtrip_serialization_preserves_signature(self, valid_domain: Domain) -> None:
        """Test that serialization and deserialization preserves spatial signature."""
        json_data = valid_domain.model_dump()
        domain2 = Domain.model_validate(json_data)
        assert domain2.spatial_signature == valid_domain.spatial_signature
