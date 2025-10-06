"""Tests for SynTriplet operation model."""

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import SynTriplet


@pytest.fixture
def valid_syn_triplet() -> SynTriplet:
    """Fixture providing a valid SynTriplet."""
    return SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")


class TestSynTripletValidation:
    """Test SynTriplet model validation."""

    def test_valid_syn_triplet_creation(self) -> None:
        """Test creating a valid SynTriplet."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        assert triplet.kind == "syn"
        assert triplet.affine == "0GenericAffine.mat"
        assert triplet.warp == "1Warp.nii.gz"
        assert triplet.inverse_warp == "1InverseWarp.nii.gz"

    def test_syn_triplet_with_paths(self) -> None:
        """Test SynTriplet with full file paths."""
        triplet = SynTriplet(
            affine="/path/to/0GenericAffine.mat",
            warp="/path/to/1Warp.nii.gz",
            inverse_warp="/path/to/1InverseWarp.nii.gz",
        )
        assert triplet.affine == "/path/to/0GenericAffine.mat"
        assert triplet.warp == "/path/to/1Warp.nii.gz"
        assert triplet.inverse_warp == "/path/to/1InverseWarp.nii.gz"

    def test_syn_triplet_rejects_empty_affine(self) -> None:
        """Test that SynTriplet rejects empty affine field."""
        with pytest.raises(ValidationError, match="affine, warp, and inverse_warp are required"):
            SynTriplet(affine="", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")

    def test_syn_triplet_rejects_empty_warp(self) -> None:
        """Test that SynTriplet rejects empty warp field."""
        with pytest.raises(ValidationError, match="affine, warp, and inverse_warp are required"):
            SynTriplet(affine="0GenericAffine.mat", warp="", inverse_warp="1InverseWarp.nii.gz")

    def test_syn_triplet_rejects_empty_inverse_warp(self) -> None:
        """Test that SynTriplet rejects empty inverse_warp field."""
        with pytest.raises(ValidationError, match="affine, warp, and inverse_warp are required"):
            SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="")

    def test_syn_triplet_kind_is_literal(self) -> None:
        """Test that kind field is correctly set to 'syn'."""
        triplet = SynTriplet(affine="0GenericAffine.mat", warp="1Warp.nii.gz", inverse_warp="1InverseWarp.nii.gz")
        assert triplet.kind == "syn"


class TestSynTripletChains:
    """Test SynTriplet chain generation."""

    def test_forward_chain_structure(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that forward_chain returns correct structure."""
        chain = valid_syn_triplet.forward_chain()
        assert chain.order == "top_to_bottom"
        assert len(chain.steps) == 2

    def test_forward_chain_step_order(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that forward_chain returns steps in correct order (warp -> affine)."""
        chain = valid_syn_triplet.forward_chain()

        # First step should be the warp field
        assert chain.steps[0].kind == "displacement_field"
        assert chain.steps[0].file == "1Warp.nii.gz"
        assert chain.steps[0].role == "forward"

        # Second step should be the affine
        assert chain.steps[1].kind == "affine"
        assert chain.steps[1].file == "0GenericAffine.mat"
        assert chain.steps[1].invert is False

    def test_inverse_chain_structure(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that inverse_chain returns correct structure."""
        chain = valid_syn_triplet.inverse_chain()
        assert chain.order == "top_to_bottom"
        assert len(chain.steps) == 2

    def test_inverse_chain_step_order(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that inverse_chain returns steps in correct order (affine inverted -> inverse warp)."""
        chain = valid_syn_triplet.inverse_chain()

        # First step should be the inverted affine
        assert chain.steps[0].kind == "affine"
        assert chain.steps[0].file == "0GenericAffine.mat"
        assert chain.steps[0].invert is True

        # Second step should be the inverse warp field
        assert chain.steps[1].kind == "displacement_field"
        assert chain.steps[1].file == "1InverseWarp.nii.gz"
        assert chain.steps[1].role == "inverse"

    def test_forward_chain_antspy_args(self, valid_syn_triplet: SynTriplet) -> None:
        """Test forward chain produces correct ANTsPy arguments."""
        chain = valid_syn_triplet.forward_chain()
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["1Warp.nii.gz", "0GenericAffine.mat"]
        assert whichtoinvert == [False, False]

    def test_inverse_chain_antspy_args(self, valid_syn_triplet: SynTriplet) -> None:
        """Test inverse chain produces correct ANTsPy arguments."""
        chain = valid_syn_triplet.inverse_chain()
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["0GenericAffine.mat", "1InverseWarp.nii.gz"]
        assert whichtoinvert == [True, False]

    def test_forward_and_inverse_chains_are_different(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that forward and inverse chains are different."""
        forward = valid_syn_triplet.forward_chain()
        inverse = valid_syn_triplet.inverse_chain()

        assert forward.steps != inverse.steps

    def test_forward_chain_uses_warp_not_inverse_warp(self) -> None:
        """Test that forward chain uses warp field, not inverse warp."""
        triplet = SynTriplet(
            affine="0GenericAffine.mat",
            warp="forward_warp.nii.gz",
            inverse_warp="inverse_warp.nii.gz",
        )
        chain = triplet.forward_chain()

        warp_step = chain.steps[0]
        assert warp_step.file == "forward_warp.nii.gz"
        assert warp_step.role == "forward"

    def test_inverse_chain_uses_inverse_warp_not_warp(self) -> None:
        """Test that inverse chain uses inverse warp field, not forward warp."""
        triplet = SynTriplet(
            affine="0GenericAffine.mat",
            warp="forward_warp.nii.gz",
            inverse_warp="inverse_warp.nii.gz",
        )
        chain = triplet.inverse_chain()

        warp_step = chain.steps[1]
        assert warp_step.file == "inverse_warp.nii.gz"
        assert warp_step.role == "inverse"

    def test_syn_triplet_roundtrip_serialization(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that SynTriplet can be serialized and deserialized."""
        json_data = valid_syn_triplet.model_dump()
        triplet2 = SynTriplet.model_validate(json_data)

        assert triplet2.kind == valid_syn_triplet.kind
        assert triplet2.affine == valid_syn_triplet.affine
        assert triplet2.warp == valid_syn_triplet.warp
        assert triplet2.inverse_warp == valid_syn_triplet.inverse_warp

    def test_forward_chain_grid_is_fixed(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that forward chain displacement field has grid='fixed'."""
        chain = valid_syn_triplet.forward_chain()
        warp_step = chain.steps[0]
        assert warp_step.grid == "fixed"

    def test_inverse_chain_grid_is_fixed(self, valid_syn_triplet: SynTriplet) -> None:
        """Test that inverse chain displacement field has grid='fixed'."""
        chain = valid_syn_triplet.inverse_chain()
        warp_step = chain.steps[1]
        assert warp_step.grid == "fixed"
