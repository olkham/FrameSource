# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `WebcamCapture` gains `backend` and `fourcc` constructor keyword arguments.
  `backend` selects the OpenCV capture backend by name (`'msmf'`, `'dshow'`,
  `'v4l2'`, `'avfoundation'`, `'gstreamer'`, `'ffmpeg'`, `'any'`) or a raw
  `cv2.CAP_*` int; `fourcc` (e.g. `'MJPG'`) requests a pixel format, applied
  before the resolution so drivers don't reset it. Fixes uncompressed formats
  (e.g. YUY2) saturating the USB link and throttling high-resolution capture
  to single-digit frame rates. `connect()` now logs a warning when it detects
  an uncompressed format negotiated at 720p or above, and `get_fourcc()`
  reports the negotiated format.

### Fixed

- Importing `framesource` now disables OpenCV MSMF hardware transforms on
  Windows (`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`, via `setdefault` so
  an explicit user value wins). This cuts webcam open time on affected devices
  from 20+ seconds to a fraction of a second with no throughput cost. Only
  effective when `framesource` is imported before `cv2` opens an MSMF device.

## [0.3.0] - 2026-07-12

The "synchronous core" release. The package was renamed, the internal
threading model was removed in favour of explicit, opt-in concurrency
helpers, and frames now carry metadata. Existing code keeps working: the
old import name, factory parameter names, and the OpenCV-compatible
`isOpened()` / `read()` / `release()` surface are all preserved.

### Added

- `Frame` — an `np.ndarray` subclass returned by `read()`, carrying
  `timestamp` (wall clock), `monotonic` (for latency/FPS math), `count`,
  `uuid`, `source`, and a free-form `metadata` dict. Works in any
  OpenCV/numpy call and survives pickling.
- `FrameSourceProtocol` — runtime-checkable protocol defining the minimal
  source contract (`connect`/`disconnect`/`read`/`is_open`), so custom
  sources can be duck-typed without inheriting `VideoCaptureBase`.
- `framesource.errors` — `FrameSourceError` base plus
  `MissingDependencyError` (an `ImportError` carrying the pip-extra install
  hint), `UnknownSourceTypeError` (a `ValueError`, raised by the factory),
  and `NotConnectedError`. Existing `except ImportError/ValueError` clauses
  keep working.
- `DeviceInfo` — typed, dict-compatible device descriptor returned by
  `discover()` implementations.
- Typed constructors with explicit keyword parameters for the consumer
  sources (`WebcamCapture`, `VideoFileCapture`, `FolderCapture`,
  `IPCameraCapture`, `ScreenCapture`, `AudioSpectrogramCapture`) and the
  vendor sources (`BaslerCapture`, `GenicamCapture`, `RealsenseCapture`,
  `XimeaCapture`, `HuatengCapture`).
- Capability flags on every source (`supports_exposure`, `supports_gain`,
  `supports_depth`) replacing `hasattr()` probing.
- Iterator protocol — `for frame in cap:` yields frames until the source is
  exhausted.
- `wait_until_ready(timeout)` — poll until a source actually produces
  frames (RTSP "connected but not yet streaming").
- `fps_actual()` — rolling-window measured FPS on the monotonic clock.
- `framesource.threading_utils` additions: `AsyncFrameSource` (asyncio
  adapter), `SharedProducer` (one producer, N subscriber queues),
  `ProducerConsumer` (joinable producer/consumer pair handle), and
  `FrameProducer.get_stats()` now reports `avg_latency` and warns on high
  drop rates.
- `FrameSourceFactory.from_config()` — create a source from a dict or a
  `.json`/`.yaml` config file (YAML requires PyYAML to be installed).
- Optional-dependency extras: `audio`, `basler`, `realsense`, `genicam`,
  `full`. `py.typed` marker shipped.
- Real-world examples: inference pipeline with back-pressure, record →
  replay, and source swapping; CI (GitHub Actions) across
  Ubuntu/Windows × Python 3.9/3.11/3.13; release automation via PyPI
  trusted publishing.

### Changed

- **Package renamed** to `framesource` (import and PyPI name now match),
  with a `src/` layout and `sources`/`processors` subpackages.
- **Synchronous core**: capture classes no longer spawn internal threads.
  Concurrency is explicit via `framesource.threading_utils`
  (`simple_frame_producer`, `FrameProducer`, multiprocessing helper).
  The sole exception is `AudioSpectrogramCapture`'s audio ring-buffer
  thread, which is inherent to its I/O model.
- Factory parameters renamed for clarity: `capture_type` → `source_type`,
  `source` → `source_id` (old names still work, see Deprecated).
- Library no longer calls `logging.basicConfig()` or prints to stdout; all
  diagnostics go through per-module loggers.
- Python floor raised to 3.9 (3.8 is end-of-life).
- Versioning is tag-driven via `setuptools_scm`.
- Lint/format tooling consolidated on ruff (replacing black + isort).

### Deprecated

- `import frame_source` — use `import framesource`. The shim (including
  deep imports like `frame_source.webcam_capture`) warns and will be
  removed in a future major release.
- Factory params `capture_type=` / `source=` — use `source_type` /
  `source_id`. Warn via `DeprecationWarning`; no removal scheduled within
  0.x.
- `get_config_schema()` on all sources — UI form schemas belong in the
  consuming application; typed constructors expose the same information.
  Scheduled for removal in 0.4.0.

### Removed

- Web-UI residue: `get_available_sources()` and the `display_fields` class
  attributes.
- Internal threading API: `start_async()`, `stop()`, `get_latest_frame()`,
  `_background_capture` (deprecated behaviourally by the synchronous core;
  the old methods no longer exist).

### Fixed

- `AudioSpectrogramCapture` accepts `freq_range` as a tuple/list as
  documented, not only as a `"min,max"` string.
- `FrameSourceFactory.create(connect=False)` no longer leaks the `connect`
  kwarg into the capture's config.
- `multiprocess_frame_producer` no longer raises `UnboundLocalError` when
  source creation fails; queue-full handling is explicit.
- `readme.md` renamed to `README.md`, fixing sdist builds on
  case-sensitive filesystems.

[Unreleased]: https://github.com/olkham/FrameSource/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/olkham/FrameSource/releases/tag/v0.3.0
