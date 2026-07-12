#!/usr/bin/env python3
"""
Source Swap Example — one processing loop, any source.

This is the library's core pitch in a single file: the exact same
``process(cap)`` loop runs unchanged over a webcam, a video file, or a folder
of images. Only ONE line changes between sources — the factory call in
``make_source()``. Everything downstream is source-agnostic because every
FrameSource capture speaks the same ``read()`` / iteration / context-manager
interface and yields the same ``Frame`` objects.

Pick the source with ``--source``:

    python source_swap_example.py --source video    # default, hardware-free
    python source_swap_example.py --source folder    # hardware-free
    python source_swap_example.py --source webcam    # REQUIRES a real camera

The ``video`` and ``folder`` sources use the bundled ``media/`` assets (a
generated clip is used if the video asset is missing), so the default run needs
no hardware. The ``webcam`` source opens device 0 and is documented as the one
path that needs real hardware.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from framesource import FrameSourceFactory


MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"


def _generate_clip(path: Path, n: int = 120, size=(640, 360), fps: int = 30) -> Path:
    """Write a tiny synthetic clip (moving shapes) as a self-contained fallback."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not open a cv2.VideoWriter to generate a demo clip")
    w, h = size
    for i in range(n):
        img = np.full((h, w, 3), 25, np.uint8)
        x = int((i / n) * (w - 80))
        cv2.rectangle(img, (x, 150), (x + 80, 230), (0, 200, 255), -1)
        cv2.circle(img, (w // 2, 40 + (i * 3) % (h - 80)), 24, (255, 120, 0), -1)
        writer.write(img)
    writer.release()
    return path


def demo_video() -> str:
    """Return a usable demo video path, preferring the bundled media asset."""
    asset = MEDIA_DIR / "geti_demo.mp4"
    if asset.exists():
        return str(asset)
    tmp = Path(__file__).resolve().parent / "_generated_demo.mp4"
    print(f"Bundled media not found; generating synthetic clip at {tmp}")
    return str(_generate_clip(tmp))


def make_source(args):
    """THE ONLY SOURCE-SPECIFIC CODE. Swapping sources swaps just this line."""
    if args.source == "webcam":
        # Requires real hardware (a camera at the given device index). On a
        # headless machine this simply fails to open and we report it.
        return FrameSourceFactory.create("webcam", source_id=args.device)
    if args.source == "folder":
        return FrameSourceFactory.create(
            "folder", source_id=str(MEDIA_DIR / "image_seq"),
            real_time=False, loop=False, watch_folder=False,
        )
    # default: a video file, paced as fast as possible for a quick demo
    return FrameSourceFactory.create(
        "video_file", source_id=demo_video(), real_time=False, loop=False,
    )


def process(cap, max_frames: int, show: bool = False):
    """Identical processing loop for EVERY source: grayscale + edge count.

    Nothing here knows whether the frames came from a webcam, a video file, or
    a folder of images. It just iterates the capture (context-manager +
    iteration protocol) and reads ``Frame`` objects with their metadata intact.
    """
    total_edges = 0
    frames = 0
    for frame in cap:  # works for any FrameSource capture
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        total_edges += int(np.count_nonzero(edges))
        frames += 1
        if frames <= 3 or frames % 20 == 0:
            print(f"  frame #{frame.count:04d}  source={frame.source}  "
                  f"edge_px={int(np.count_nonzero(edges))}")
        if show:
            try:
                cv2.imshow("source_swap", edges)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass  # headless: no GUI backend, keep processing
        if frames >= max_frames:
            break
    avg = (total_edges / frames) if frames else 0.0
    return frames, avg


def run(args) -> None:
    print(f"Source: {args.source}")
    cap = make_source(args)
    if not cap.is_open():
        if args.source == "webcam":
            print("Could not open the webcam (device "
                  f"{args.device}). This source needs real camera hardware.")
        else:
            print("Failed to open the source.")
        return

    # Context manager handles disconnect; the loop below is fully source-agnostic.
    with cap:
        frames, avg_edges = process(cap, args.frames, show=args.show)

    print("\n=== Summary ===")
    print(f"  source        : {args.source}")
    print(f"  frames read   : {frames}")
    print(f"  avg edge px   : {avg_edges:.0f} per frame")
    print("  (the process() loop above was byte-for-byte identical across sources)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["webcam", "video", "folder"],
                        default="video", help="frame source to run the SAME loop over")
    parser.add_argument("--frames", type=int, default=60, help="max frames to process")
    parser.add_argument("--device", type=int, default=0, help="webcam device index (webcam source only)")
    parser.add_argument("--show", action="store_true", help="display edges (off by default for headless use)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
