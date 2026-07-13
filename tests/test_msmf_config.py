"""Verify importing framesource configures the MSMF backend (Windows only).

The setting is read by OpenCV at ``import cv2`` time, so these run in fresh
subprocesses with a controlled environment rather than relying on the state
of the current interpreter.
"""

import os
import platform
import subprocess
import sys

import pytest

_VAR = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"


def _import_framesource_with_env(overrides):
    """Import framesource in a clean subprocess; return the var's value ('None' if unset)."""
    env = dict(os.environ)
    env.pop(_VAR, None)
    env.update(overrides)
    code = f"import os, framesource; print(os.environ.get({_VAR!r}))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(platform.system() != "Windows", reason="MSMF config only applies on Windows")
def test_sets_default_on_windows():
    assert _import_framesource_with_env({}) == "0"


@pytest.mark.skipif(platform.system() != "Windows", reason="MSMF config only applies on Windows")
def test_respects_explicit_user_value():
    # setdefault must not clobber a value the user set deliberately.
    assert _import_framesource_with_env({_VAR: "1"}) == "1"


@pytest.mark.skipif(platform.system() == "Windows", reason="no-op off Windows")
def test_noop_off_windows():
    assert _import_framesource_with_env({}) == "None"
