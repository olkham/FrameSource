#!/usr/bin/env python3
"""
Inference Pipeline Example — a slow model on a live stream.

This is THE recipe for "my model is slower than my camera". A background
``FrameProducer`` keeps the source drained at capture rate while the main
thread runs a deliberately slow, stubbed "model" at its own (slower) pace.

What it demonstrates:
  * Queue back-pressure with a small ``max_queue_size``: because the producer
    enqueues with ``put(block=False)`` and drops when the queue is full, a slow
    consumer never stalls the capture — the newest frames are simply dropped.
  * Reading from the queue at *model* pace while the producer keeps reading the
    source at *capture* pace.
  * ``FrameProducer.get_stats()`` at the end, highlighting ``frames_dropped``
    (the price of a slow model) and ``avg_latency`` (producer-side enqueue lag).
  * A consumer-side end-to-end latency (``now - frame.timestamp``) that grows
    with queue depth — the tell-tale sign the queue is holding stale frames.

Runs hardware-free using ``media/geti_demo.mp4`` (looped to simulate a live
feed); if that asset is missing a tiny synthetic clip is generated at startup.
The stub model is cv2-only (blur -> Otsu threshold -> contour count) so no ML
dependency is required; swap its body for a real forward pass and nothing else
in the pipeline changes.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from framesource import FrameSourceFactory
from framesource.threading_utils import FrameProducer

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


def stub_model(frame: np.ndarray, infer_delay: float) -> int:
    """Placeholder for a slow deep-learning model (cv2-only, no ML deps).

    Simulates a detector: downscale -> blur -> Otsu threshold -> contour count,
    with a deliberate sleep so the "model" runs slower than the incoming frame
    rate. Replace the body with a real forward pass; the pipeline is unchanged.
    """
    small = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    time.sleep(infer_delay)  # simulate slow inference
    return len(contours)


def run(args) -> None:
    """Feed a slow model from a live-paced source and report drop/latency stats."""
    # real_time + loop makes the finite clip behave like an endless live camera
    # delivering frames at its native rate, independent of how fast we consume.
    cap = FrameSourceFactory.create("video_file", source_id=demo_video(), real_time=True, loop=True)
    if not cap.is_open():
        print("Failed to open the demo video source.")
        return

    # A SMALL queue is the whole point: it forces back-pressure to show up fast.
    producer = FrameProducer(cap, max_queue_size=args.queue, target_fps=None)
    producer.start()

    processed = 0
    e2e_latencies = []  # consumer-side: capture timestamp -> model start
    empty_reads = 0
    print(
        f"Consuming {args.frames} model inferences "
        f"(queue={args.queue}, infer_delay={args.infer_delay:.3f}s)...\n"
    )
    try:
        while processed < args.frames:
            ok, frame = producer.get_frame(timeout=1.0)
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 5:  # producer never delivered — bail rather than hang
                    print("Producer stopped delivering frames; stopping early.")
                    break
                continue
            empty_reads = 0

            # End-to-end latency: how stale is this frame by the time the model
            # sees it? Under back-pressure this trends toward queue_depth / fps.
            e2e_latencies.append(time.time() - frame.timestamp)
            n_contours = stub_model(frame, args.infer_delay)
            processed += 1
            if processed <= 3 or processed % 10 == 0:
                print(
                    f"  inference #{processed:03d}  frame.count={frame.count:<4}  "
                    f"contours={n_contours:<3}  stale={e2e_latencies[-1] * 1000:6.1f} ms"
                )
    finally:
        producer.stop()

    stats = producer.get_stats()
    captured = stats.get("frames_captured", 0)
    dropped = stats.get("frames_dropped", 0)
    total = captured + dropped
    drop_pct = (100.0 * dropped / total) if total else 0.0
    avg_lat = stats.get("avg_latency")
    mean_e2e = (sum(e2e_latencies) / len(e2e_latencies)) if e2e_latencies else 0.0

    print("\n=== FrameProducer.get_stats() ===")
    print(f"  frames_captured : {captured}")
    print(f"  frames_dropped  : {dropped}   <-- dropped because the model is slow")
    print(f"  drop rate       : {drop_pct:.1f}% of captured frames")
    print(f"  producer fps    : {stats.get('fps', 0.0):.1f}")
    avg_lat_ms = f"{avg_lat * 1000:.2f} ms" if avg_lat is not None else "n/a"
    print(f"  avg_latency     : {avg_lat_ms}  (producer stamp -> enqueue)")
    print("\n=== Consumer view ===")
    print(f"  inferences run  : {processed}")
    print(
        f"  mean staleness  : {mean_e2e * 1000:.1f} ms  (capture -> model, grows with queue depth)"
    )
    if drop_pct > 10:
        print(
            "\nTakeaway: the model can't keep up, so newest frames are dropped and "
            "the consumer always works on recent (not backlogged) data. Increase "
            "max_queue_size to buffer more, or speed up the model to drop fewer."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=int, default=40, help="model inferences to run before stopping"
    )
    parser.add_argument(
        "--queue", type=int, default=4, help="max_queue_size (small => visible back-pressure)"
    )
    parser.add_argument(
        "--infer-delay", type=float, default=0.08, help="simulated per-frame inference time (s)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
