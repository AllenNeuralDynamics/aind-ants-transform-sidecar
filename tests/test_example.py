"""Example test module."""

import aind_ants_transform_sidecar


def test_version():
    """Test that version is defined."""
    assert aind_ants_transform_sidecar.__version__ is not None
    assert isinstance(aind_ants_transform_sidecar.__version__, str)
