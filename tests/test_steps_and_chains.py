"""Tests for transform steps and chains."""

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import Chain, StepAffine, StepField


class TestStepAffine:
    """Test StepAffine model."""

    def test_step_affine_creation(self) -> None:
        """Test creating a StepAffine."""
        step = StepAffine(file="0GenericAffine.mat")
        assert step.kind == "affine"
        assert step.file == "0GenericAffine.mat"
        assert step.invert is False

    def test_step_affine_with_invert(self) -> None:
        """Test creating a StepAffine with invert flag."""
        step = StepAffine(file="0GenericAffine.mat", invert=True)
        assert step.kind == "affine"
        assert step.file == "0GenericAffine.mat"
        assert step.invert is True

    def test_step_affine_with_path(self) -> None:
        """Test StepAffine with full file path."""
        step = StepAffine(file="/path/to/transforms/0GenericAffine.mat")
        assert step.file == "/path/to/transforms/0GenericAffine.mat"

    def test_step_affine_kind_is_literal(self) -> None:
        """Test that kind field is correctly set to 'affine'."""
        step = StepAffine(file="test.mat")
        assert step.kind == "affine"


class TestStepField:
    """Test StepField model."""

    def test_step_field_forward_creation(self) -> None:
        """Test creating a forward StepField."""
        step = StepField(file="1Warp.nii.gz", role="forward")
        assert step.kind == "displacement_field"
        assert step.file == "1Warp.nii.gz"
        assert step.role == "forward"
        assert step.grid == "fixed"

    def test_step_field_inverse_creation(self) -> None:
        """Test creating an inverse StepField."""
        step = StepField(file="1InverseWarp.nii.gz", role="inverse")
        assert step.kind == "displacement_field"
        assert step.file == "1InverseWarp.nii.gz"
        assert step.role == "inverse"
        assert step.grid == "fixed"

    def test_step_field_grid_defaults_to_fixed(self) -> None:
        """Test that grid defaults to 'fixed'."""
        step = StepField(file="1Warp.nii.gz", role="forward")
        assert step.grid == "fixed"

    def test_step_field_rejects_non_fixed_grid(self) -> None:
        """Test that StepField rejects non-'fixed' grid values."""
        # This test depends on how Pydantic handles Literal validation
        # The validator will catch it if somehow passed through
        with pytest.raises(ValidationError):
            # Try to create with wrong grid value - should fail at Pydantic level
            StepField(file="1Warp.nii.gz", role="forward", grid="moving")  # type: ignore[arg-type]

    def test_step_field_with_path(self) -> None:
        """Test StepField with full file path."""
        step = StepField(file="/data/warps/1Warp.nii.gz", role="forward")
        assert step.file == "/data/warps/1Warp.nii.gz"

    def test_step_field_kind_is_literal(self) -> None:
        """Test that kind field is correctly set to 'displacement_field'."""
        step = StepField(file="test.nii.gz", role="forward")
        assert step.kind == "displacement_field"


class TestChain:
    """Test Chain model."""

    def test_chain_with_single_affine(self) -> None:
        """Test Chain with single affine step."""
        steps = [StepAffine(file="0GenericAffine.mat")]
        chain = Chain(steps=steps)
        assert chain.order == "top_to_bottom"
        assert len(chain.steps) == 1
        assert chain.steps[0].kind == "affine"

    def test_chain_with_single_field(self) -> None:
        """Test Chain with single displacement field step."""
        steps = [StepField(file="1Warp.nii.gz", role="forward")]
        chain = Chain(steps=steps)
        assert len(chain.steps) == 1
        assert chain.steps[0].kind == "displacement_field"

    def test_chain_with_mixed_steps(self) -> None:
        """Test Chain with both affine and field steps."""
        steps = [
            StepField(file="1Warp.nii.gz", role="forward"),
            StepAffine(file="0GenericAffine.mat", invert=False),
        ]
        chain = Chain(steps=steps)
        assert len(chain.steps) == 2
        assert chain.steps[0].kind == "displacement_field"
        assert chain.steps[1].kind == "affine"

    def test_chain_empty_steps(self) -> None:
        """Test Chain with empty steps list."""
        chain = Chain(steps=[])
        assert len(chain.steps) == 0

    def test_chain_order_is_top_to_bottom(self) -> None:
        """Test that Chain order is 'top_to_bottom'."""
        chain = Chain(steps=[])
        assert chain.order == "top_to_bottom"

    def test_antspy_apply_transforms_args_affine_only(self) -> None:
        """Test antspy_apply_transforms_args with affine steps."""
        steps = [
            StepAffine(file="transform1.mat", invert=False),
            StepAffine(file="transform2.mat", invert=True),
        ]
        chain = Chain(steps=steps)
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["transform1.mat", "transform2.mat"]
        assert whichtoinvert == [False, True]

    def test_antspy_apply_transforms_args_field_only(self) -> None:
        """Test antspy_apply_transforms_args with displacement field steps."""
        steps = [
            StepField(file="1Warp.nii.gz", role="forward"),
            StepField(file="1InverseWarp.nii.gz", role="inverse"),
        ]
        chain = Chain(steps=steps)
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["1Warp.nii.gz", "1InverseWarp.nii.gz"]
        assert whichtoinvert == [False, False]

    def test_antspy_apply_transforms_args_mixed(self) -> None:
        """Test antspy_apply_transforms_args with mixed steps."""
        steps = [
            StepField(file="1Warp.nii.gz", role="forward"),
            StepAffine(file="0GenericAffine.mat", invert=False),
        ]
        chain = Chain(steps=steps)
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["1Warp.nii.gz", "0GenericAffine.mat"]
        assert whichtoinvert == [False, False]

    def test_antspy_apply_transforms_args_complex_chain(self) -> None:
        """Test antspy_apply_transforms_args with complex chain."""
        steps = [
            StepField(file="warp1.nii.gz", role="forward"),
            StepAffine(file="affine1.mat", invert=True),
            StepAffine(file="affine2.mat", invert=False),
            StepField(file="warp2.nii.gz", role="inverse"),
        ]
        chain = Chain(steps=steps)
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == ["warp1.nii.gz", "affine1.mat", "affine2.mat", "warp2.nii.gz"]
        assert whichtoinvert == [False, True, False, False]

    def test_antspy_apply_transforms_args_empty_chain(self) -> None:
        """Test antspy_apply_transforms_args with empty chain."""
        chain = Chain(steps=[])
        transforms, whichtoinvert = chain.antspy_apply_transforms_args()

        assert transforms == []
        assert whichtoinvert == []

    def test_chain_preserves_step_order(self) -> None:
        """Test that Chain preserves the order of steps."""
        steps = [
            StepAffine(file="step1.mat"),
            StepField(file="step2.nii.gz", role="forward"),
            StepAffine(file="step3.mat"),
        ]
        chain = Chain(steps=steps)

        assert chain.steps[0].file == "step1.mat"
        assert chain.steps[1].file == "step2.nii.gz"
        assert chain.steps[2].file == "step3.mat"

    def test_chain_roundtrip_serialization(self) -> None:
        """Test that Chain can be serialized and deserialized."""
        steps = [
            StepField(file="1Warp.nii.gz", role="forward"),
            StepAffine(file="0GenericAffine.mat", invert=True),
        ]
        chain = Chain(steps=steps)

        json_data = chain.model_dump()
        chain2 = Chain.model_validate(json_data)

        assert chain2.steps[0].kind == "displacement_field"
        assert chain2.steps[0].file == "1Warp.nii.gz"
        assert chain2.steps[1].kind == "affine"
        assert chain2.steps[1].file == "0GenericAffine.mat"
        assert chain2.steps[1].invert is True
