"""Tests for BBox model."""

import math

import pytest
from pydantic import ValidationError

from aind_ants_transform_sidecar.sidecar import BBox


class TestBBoxValidation:
    """Test BBox validation rules."""

    def test_valid_bbox_creation(self) -> None:
        """Test creating a valid BBox."""
        bbox = BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))
        assert bbox.L == (-50.0, 50.0)
        assert bbox.P == (-60.0, 40.0)
        assert bbox.S == (-30.0, 70.0)

    def test_bbox_min_equals_max(self) -> None:
        """Test BBox where min equals max (single point)."""
        bbox = BBox(L=(0.0, 0.0), P=(5.0, 5.0), S=(-10.0, -10.0))
        assert bbox.L == (0.0, 0.0)
        assert bbox.P == (5.0, 5.0)
        assert bbox.S == (-10.0, -10.0)

    def test_bbox_rejects_non_monotonic_L(self) -> None:
        """Test that BBox rejects L axis where min > max."""
        with pytest.raises(ValidationError, match="bbox\\[L\\] min must be ≤ max"):
            BBox(L=(50.0, -50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))

    def test_bbox_rejects_non_monotonic_P(self) -> None:
        """Test that BBox rejects P axis where min > max."""
        with pytest.raises(ValidationError, match="bbox\\[P\\] min must be ≤ max"):
            BBox(L=(-50.0, 50.0), P=(40.0, -60.0), S=(-30.0, 70.0))

    def test_bbox_rejects_non_monotonic_S(self) -> None:
        """Test that BBox rejects S axis where min > max."""
        with pytest.raises(ValidationError, match="bbox\\[S\\] min must be ≤ max"):
            BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(70.0, -30.0))

    def test_bbox_rejects_inf_in_L(self) -> None:
        """Test that BBox rejects infinity in L axis."""
        # inf > any finite number, so it fails monotonic check first
        with pytest.raises(ValidationError, match="bbox\\[L\\]"):
            BBox(L=(math.inf, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))

        with pytest.raises(ValidationError, match="bbox\\[L\\] contains non-finite value"):
            BBox(L=(-50.0, math.inf), P=(-60.0, 40.0), S=(-30.0, 70.0))

    def test_bbox_rejects_inf_in_P(self) -> None:
        """Test that BBox rejects infinity in P axis."""
        with pytest.raises(ValidationError, match="bbox\\[P\\] contains non-finite value"):
            BBox(L=(-50.0, 50.0), P=(-math.inf, 40.0), S=(-30.0, 70.0))

    def test_bbox_rejects_inf_in_S(self) -> None:
        """Test that BBox rejects infinity in S axis."""
        with pytest.raises(ValidationError, match="bbox\\[S\\] contains non-finite value"):
            BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(-30.0, math.inf))

    def test_bbox_rejects_nan_in_L(self) -> None:
        """Test that BBox rejects NaN in L axis."""
        with pytest.raises(ValidationError, match="bbox\\[L\\] contains non-finite value"):
            BBox(L=(math.nan, 50.0), P=(-60.0, 40.0), S=(-30.0, 70.0))

    def test_bbox_rejects_nan_in_P(self) -> None:
        """Test that BBox rejects NaN in P axis."""
        with pytest.raises(ValidationError, match="bbox\\[P\\] contains non-finite value"):
            BBox(L=(-50.0, 50.0), P=(math.nan, 40.0), S=(-30.0, 70.0))

    def test_bbox_rejects_nan_in_S(self) -> None:
        """Test that BBox rejects NaN in S axis."""
        with pytest.raises(ValidationError, match="bbox\\[S\\] contains non-finite value"):
            BBox(L=(-50.0, 50.0), P=(-60.0, 40.0), S=(math.nan, 70.0))

    def test_bbox_with_negative_coordinates(self) -> None:
        """Test BBox with all negative coordinates."""
        bbox = BBox(L=(-100.0, -50.0), P=(-200.0, -150.0), S=(-80.0, -20.0))
        assert bbox.L == (-100.0, -50.0)
        assert bbox.P == (-200.0, -150.0)
        assert bbox.S == (-80.0, -20.0)

    def test_bbox_with_large_values(self) -> None:
        """Test BBox with very large coordinate values."""
        bbox = BBox(L=(-1e6, 1e6), P=(-1e6, 1e6), S=(-1e6, 1e6))
        assert bbox.L == (-1e6, 1e6)
        assert bbox.P == (-1e6, 1e6)
        assert bbox.S == (-1e6, 1e6)

    def test_bbox_with_small_range(self) -> None:
        """Test BBox with very small range."""
        bbox = BBox(L=(0.0, 0.001), P=(0.0, 0.001), S=(0.0, 0.001))
        assert bbox.L == (0.0, 0.001)
        assert bbox.P == (0.0, 0.001)
        assert bbox.S == (0.0, 0.001)
