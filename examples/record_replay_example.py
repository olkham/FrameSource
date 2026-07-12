#!/usr/bin/env python3
"""
Record / Replay Example — capture a stream to disk, then play it back in step.

Two phases:

  Phase 1 (record): read N frames from any source and write each one to a
  folder using a filename that embeds the frame's ``count`` and wall-clock
  ``timestamp`` (``frame_<count>_<microseconds>.png``). Because the count is
  zero-padded, a plain lexical sort replays the frames in capture order.

  Phase 2 (replay): re-open that folder through
  ``FrameSourceFactory.create('folder', source_id=..., fps=..., real_time=True)``.
  The replay ``fps`` is derived from the *recorded* timestamps, so playback
  reproduces the original pacing. For every replayed frame we parse the
  ``count`` / ``timestamp`` back out of the filename and confirm they line up,
  frame-for-frame, with what we recorded.

Runs hardware-free using ``media/geti_demo.mp4`` (or a generated clip if that
asset is missing). Frames are captured with ``real_time=True`` so the recorded
timestamps carry genuine inter-frame spacing to reproduce. A temporary output
folder is created and cleaned up automatically.
"""

import argparse
import os
import shutil
import statistics
import tempfile
import time
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


def frame_filename(count: int, timestamp: float) -> str:
    """Encode count + timestamp into a lexically-sortable filename."""
    ts_us = int(round(timestamp * 1_000_000))
    return f"frame_{count:06d}_{ts_us}.png"


def parse_filename(path: str):
    """Recover (count, timestamp) from a filename produced by frame_filename()."""
    stem = os.path.splitext(os.path.basename(path))[0]
    _, count_s, ts_us_s = stem.split("_")
    return int(count_s), int(ts_us_s) / 1_000_000


def record_phase(src_path: str, n_frames: int, out_dir: str):
    """Capture n_frames from the video source, writing timestamped PNGs.

    Returns the list of recorded (count, timestamp) tuples.
    """
    # real_time=True so the recorded timestamps have real inter-frame spacing;
    # loop=True guarantees we can grab n_frames from even a short clip.
    cap = FrameSourceFactory.create(
        "video_file", source_id=src_path, real_time=True, loop=True
    )
    records = []
    try:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, (640, 360))  # keep the temp folder light
            fname = frame_filename(frame.count, frame.timestamp)
            cv2.imwrite(os.path.join(out_dir, fname), small)
            records.append((frame.count, frame.timestamp))
    finally:
        cap.disconnect()
    return records


def replay_phase(out_dir: str, fps: float):
    """Replay the recorded folder at the given fps, parsing metadata back.

    Returns a list of (count, timestamp, monotonic) tuples, one per replayed
    frame, where count/timestamp come from the filename and monotonic is the
    replay-side clock reading (used to measure reproduced pacing).
    """
    cap = FrameSourceFactory.create(
        "folder",
        source_id=out_dir,
        sort_by="name",       # zero-padded count => capture order
        fps=fps,              # reproduce the recorded pacing...
        real_time=True,       # ...by actually sleeping between frames
        loop=False,
        watch_folder=False,   # no watchdog dependency needed for replay
    )
    replayed = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            count, ts = parse_filename(cap.get_current_file_path())
            replayed.append((count, ts, frame.monotonic))
    finally:
        cap.disconnect()
    return replayed


def recorded_fps(records) -> float:
    """Estimate the source fps from the median gap between recorded timestamps."""
    stamps = [ts for _, ts in records]
    deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not deltas:
        return 30.0
    return 1.0 / statistics.median(deltas)


def run(args) -> None:
    out_dir = tempfile.mkdtemp(prefix="framesource_record_")
    try:
        print(f"Phase 1: recording {args.frames} frames -> {out_dir}")
        records = record_phase(demo_video(), args.frames, out_dir)
        if not records:
            print("No frames captured; aborting.")
            return
        fps = recorded_fps(records)
        print(f"  recorded {len(records)} frames; estimated source fps = {fps:.1f}\n")

        print(f"Phase 2: replaying folder at {fps:.1f} fps (reproducing pacing)")
        t0 = time.monotonic()
        replayed = replay_phase(out_dir, fps)
        wall = time.monotonic() - t0

        # Verify metadata parsed from filenames lines up with what we recorded.
        matched = 0
        for i, (count, ts, _mono) in enumerate(replayed):
            rec_count, rec_ts = records[i]
            if count == rec_count and abs(ts - rec_ts) < 1e-6:
                matched += 1

        print(f"  replayed {len(replayed)} frames in {wall:.2f}s")
        print("\n  first frames (count / timestamp round-tripped through filenames):")
        for count, ts, _mono in replayed[: min(5, len(replayed))]:
            print(f"    count={count:<4} timestamp={ts:.6f}")

        # Measure reproduced pacing from replay-side monotonic clock.
        mono = [m for _, _, m in replayed]
        rdeltas = [b - a for a, b in zip(mono, mono[1:]) if b > a]
        measured = (1.0 / statistics.median(rdeltas)) if rdeltas else float("nan")

        print("\n=== Summary ===")
        print(f"  metadata match     : {matched}/{len(replayed)} frames line up")
        print(f"  target replay fps  : {fps:.1f}")
        print(f"  measured replay fps: {measured:.1f}  (real_time pacing honored)")
        if matched == len(replayed) == len(records):
            print("  OK: every recorded frame replayed in order with matching metadata.")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=24, help="frames to record then replay")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
