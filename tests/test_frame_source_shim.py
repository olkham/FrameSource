"""Regression tests for the deprecated ``frame_source`` compatibility shim.

The shim (``src/frame_source/__init__.py``) emits a DeprecationWarning on import
and aliases the old flat module paths (e.g. ``frame_source.webcam_capture``) onto
the new ``framesource.sources.*`` locations. These tests pin that behaviour.

Importing a module is cached in ``sys.modules``, so the top-level
``warnings.warn`` only fires on the *first* import. Each test therefore purges
``frame_source`` (and its aliased submodules) from ``sys.modules`` first, via a
fixture that restores the original state afterwards so test order does not matter.
"""

import importlib
import sys

import pytest


def _frame_source_module_names():
    """Names of the shim package and any aliased submodules in sys.modules."""
    return [
        name
        for name in list(sys.modules)
        if name == "frame_source" or name.startswith("frame_source.")
    ]


def _purge_frame_source_modules():
    """Remove ``frame_source`` and every aliased submodule from sys.modules."""
    for name in _frame_source_module_names():
        del sys.modules[name]


@pytest.fixture
def fresh_frame_source_import():
    """Force a fresh ``frame_source`` import, restoring sys.modules afterwards."""
    saved = {name: sys.modules[name] for name in _frame_source_module_names()}
    _purge_frame_source_modules()
    try:
        yield
    finally:
        _purge_frame_source_modules()
        sys.modules.update(saved)


def test_import_emits_deprecation_warning(fresh_frame_source_import):
    with pytest.warns(DeprecationWarning, match="frame_source"):
        importlib.import_module("frame_source")


def test_factory_is_the_same_object(fresh_frame_source_import):
    with pytest.warns(DeprecationWarning):
        frame_source = importlib.import_module("frame_source")
    import framesource

    assert frame_source.FrameSourceFactory is framesource.FrameSourceFactory


def test_deep_submodule_import_resolves(fresh_frame_source_import):
    with pytest.warns(DeprecationWarning):
        importlib.import_module("frame_source")

    module = importlib.import_module("frame_source.webcam_capture")
    import framesource.sources.webcam_capture as canonical

    assert module is canonical
    assert module.WebcamCapture is canonical.WebcamCapture
