"""Configure OpenCV's Media Foundation (MSMF) backend before ``cv2`` loads.

On Windows, OpenCV's MSMF capture backend reads
``OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS`` **once, when cv2 is first
imported**. With hardware transforms enabled (the default), opening some
webcams stalls for 20+ seconds while Media Foundation initializes hardware
transforms — even though frame throughput afterwards is fine. Disabling them
makes the camera open near-instantly (measured 22.6s -> 0.33s) with no
measurable throughput cost.

Because the value is read at ``import cv2`` time, it must be set *before* cv2
is imported anywhere in the process. This module carries no cv2 dependency and
is imported first by :mod:`framesource`, ahead of any capture module, so the
setting is in place in time. ``setdefault`` is used so a user who configured
the variable explicitly (to ``"0"`` or ``"1"``) always wins.

Caveat: if the host application runs ``import cv2`` and opens an MSMF device
*before* importing :mod:`framesource`, cv2 has already cached the value and
this has no effect; set the environment variable yourself in that case.
"""

import os
import platform

if platform.system() == "Windows":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
