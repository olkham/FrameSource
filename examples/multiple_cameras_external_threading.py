#!/usr/bin/env python3
"""
Multiple Cameras Example with External Threading

Demonstrates connecting to multiple different camera types simultaneously using
external threading. Each source has its own producer thread; the main thread is
a pure display loop.

The key to good multi-camera performance is that the display loop must never
block on one source. Two rules make it smooth:

1. Drain each camera's queue non-blockingly to the *latest* frame and discard
   the backlog. A FIFO queue otherwise hands you the oldest frame, so any
   moment the consumer falls behind becomes permanent latency.
2. Never call a blocking ``queue.get(timeout=...)`` per camera in the shared
   loop: a slow or stalled stream (e.g. a laggy IP camera) would hold up every
   other camera for the length of the timeout, starving fast cameras.

With these, a fast webcam stays low-latency and near its native frame rate even
alongside a slow IP stream.
"""

import queue
import threading
import time
from typing import Any


from framesource import FrameSourceFactory
from framesource.threading_utils import simple_frame_producer
import cv2


def main():
    """Test multiple cameras with external threading."""
    print("Testing Multiple Cameras with External Threading:")

    # Configuration for different camera types. ``target_fps`` controls the
    # producer's pacing: None means "as fast as the source delivers" (the
    # webcam runs at its native 60fps; the IP camera self-paces on the network).
    cameras_config: list[dict[str, Any]] = [
        {
            "source_type": "webcam",
            "source_id": 2,
            "width": 1920,
            "height": 1080,
            # Request your camera's supported rate. Many 1080p USB webcams cap
            # at 30fps regardless of what you ask for — requesting more just
            # yields the camera's maximum. target_fps=None below then lets the
            # producer run at whatever the camera actually delivers.
            "fps": 30,
            "backend": "msmf",
            "fourcc": "MJPG",
            "target_fps": None,
        },
        {
            "source_type": "ipcam",
            "source_id": "https://195.196.36.242/mjpg/video.mjpg",
            "target_fps": None,
        },
        # {'source_type': 'video_file', 'source_id': "media/geti_demo.mp4", 'loop': True},
        # {'source_type': 'screen', 'x': 100, 'y': 100, 'w': 400, 'h': 300, 'fps': 15},
        # {'source_type': 'realsense', 'width': 1280, 'height': 720},
    ]

    camera_systems = []  # List of (name, camera, queue, stop_event, thread)
    grid_cols = 2
    win_w, win_h = 640, 480

    # Connect to all cameras and start their producers
    for idx, cam_cfg in enumerate(cameras_config):
        cfg = dict(cam_cfg)  # don't mutate the source config
        name = cfg.pop("source_type", None)
        if not name:
            print(f"Camera config missing 'source_type': {cam_cfg}")
            continue
        target_fps = cfg.pop("target_fps", None)

        print(f"Setting up {name}...")

        try:
            # Create the source without auto-connecting so we connect exactly
            # once, then decide whether to start a producer for it.
            camera = FrameSourceFactory.create(name, connect=False, **cfg)

            if camera.connect():
                # Configure window position in grid
                cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"{name}", win_w, win_h)
                col = idx % grid_cols
                row = idx // grid_cols
                x = col * (win_w + 10)
                y = row * (win_h + 50)
                cv2.moveWindow(f"{name}", x, y)

                # Enable auto exposure if supported
                try:
                    camera.enable_auto_exposure(True)
                except Exception:
                    pass

                # A small queue is enough: the consumer keeps only the latest
                # frame, so there is no reason to buffer a deep backlog.
                frame_queue: queue.Queue = queue.Queue(maxsize=2)
                stop_event = threading.Event()

                producer_thread = threading.Thread(
                    target=simple_frame_producer,
                    args=(camera, frame_queue, stop_event, target_fps),
                    daemon=True,
                )
                producer_thread.start()

                camera_systems.append((name, camera, frame_queue, stop_event, producer_thread))
                print(f"[ok] {name} producer started")
            else:
                print(f"[fail] Failed to connect to {name}")
        except Exception as e:
            print(f"[fail] Error setting up {name}: {e}")

    if not camera_systems:
        print("No cameras set up successfully")
        return

    print(f"\nSet up {len(camera_systems)} camera producers")
    print("Press 'q' to quit, 'h' for help")

    def print_help():
        print("\nMultiple Cameras Controls:")
        print("  q - Quit")
        print("  h - Show this help")
        print("  s - Show statistics (fps and latency per camera)")

    # Per-camera counters for a live fps/latency readout.
    frame_counts = {name: 0 for name, _, _, _, _ in camera_systems}
    fps_window = {name: [] for name, _, _, _, _ in camera_systems}  # recent frame timestamps
    last_latency = {name: 0.0 for name, _, _, _, _ in camera_systems}  # seconds
    last_status_time = time.time()

    def latest_frame(frame_queue):
        """Non-blocking: drain the queue and return only the most recent item."""
        latest = None
        try:
            while True:
                latest = frame_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    try:
        while True:
            any_alive = False

            for name, _camera, frame_queue, stop_event, thread in camera_systems:
                if stop_event.is_set() or not thread.is_alive():
                    continue
                any_alive = True

                item = latest_frame(frame_queue)
                if item is None:
                    continue  # no new frame this tick — do not block

                success, frame = item
                if success and frame is not None:
                    cv2.imshow(f"{name}", frame)
                    frame_counts[name] += 1
                    now = time.time()
                    last_latency[name] = now - float(getattr(frame, "timestamp", now))
                    window = fps_window[name]
                    window.append(now)
                    if len(window) > 30:
                        del window[0]

            if not any_alive:
                print("All camera producers stopped")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                print_help()
            elif key == ord("s"):
                _print_stats(camera_systems, frame_counts, fps_window, last_latency)

            # Print status every 5 seconds
            if time.time() - last_status_time > 5.0:
                _print_stats(camera_systems, frame_counts, fps_window, last_latency)
                last_status_time = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("Stopping camera producers...")
        for name, _camera, _frame_queue, stop_event, thread in camera_systems:
            try:
                print(f"Stopping {name}...")
                stop_event.set()
                thread.join(timeout=2)
                print(f"[ok] {name} stopped cleanly")
            except Exception as e:
                print(f"[fail] Error stopping {name}: {e}")

        cv2.destroyAllWindows()

        # Final statistics
        total_frames = sum(frame_counts.values())
        print("\nFinal statistics:")
        for name, count in frame_counts.items():
            print(f"  {name}: {count} frames")
        print(f"  Total: {total_frames} frames")


def _measure_fps(timestamps):
    """Rolling fps from a list of recent wall-clock display times."""
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / span if span > 0 else 0.0


def _print_stats(camera_systems, frame_counts, fps_window, last_latency):
    parts = []
    for name, _c, _q, _s, _t in camera_systems:
        fps = _measure_fps(fps_window[name])
        parts.append(
            f"{name}: {frame_counts[name]} frames, "
            f"{fps:.1f} fps, {1000 * last_latency[name]:.0f} ms latency"
        )
    print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
