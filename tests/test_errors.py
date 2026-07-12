"""Tests for the framesource.errors exception hierarchy (hardware-free)."""

import pytest

from framesource.errors import (
    FrameSourceError,
    MissingDependencyError,
    UnknownSourceTypeError,
    NotConnectedError,
)
from framesource import FrameSourceFactory


def test_missing_dependency_error_is_import_error():
    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(MissingDependencyError, FrameSourceError)


def test_unknown_source_type_error_is_value_error():
    assert issubclass(UnknownSourceTypeError, ValueError)
    assert issubclass(UnknownSourceTypeError, FrameSourceError)


def test_not_connected_error_is_frame_source_error():
    assert issubclass(NotConnectedError, FrameSourceError)


def test_all_errors_derive_from_frame_source_error():
    for exc_type in (MissingDependencyError, UnknownSourceTypeError, NotConnectedError):
        assert issubclass(exc_type, FrameSourceError)
    assert issubclass(FrameSourceError, RuntimeError)


def test_missing_dependency_error_message_contains_package_and_install_hint():
    err = MissingDependencyError('pyaudio', extra='audio')
    message = str(err)
    assert 'pyaudio' in message
    assert 'pip install framesource[audio]' in message


def test_missing_dependency_error_without_extra_omits_hint():
    err = MissingDependencyError('somepkg')
    message = str(err)
    assert 'somepkg' in message
    assert 'pip install' not in message


def test_missing_dependency_error_includes_details():
    err = MissingDependencyError('librosa/soundfile/pyaudio', extra='audio', details='No module named librosa')
    message = str(err)
    assert 'No module named librosa' in message


def test_missing_dependency_error_stores_attributes():
    err = MissingDependencyError('pyaudio', extra='audio')
    assert err.package == 'pyaudio'
    assert err.extra == 'audio'


def test_factory_unknown_type_raises_unknown_source_type_error():
    with pytest.raises(UnknownSourceTypeError):
        FrameSourceFactory.create('definitely_not_a_type', source_id=0)


def test_factory_unknown_type_still_caught_by_value_error():
    # Backwards compatibility: old user code catching plain ValueError
    # must keep working.
    with pytest.raises(ValueError):
        FrameSourceFactory.create('definitely_not_a_type', source_id=0)
