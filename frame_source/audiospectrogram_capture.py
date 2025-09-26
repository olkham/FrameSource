import numpy as np
import cv2
import threading
import time
import logging
from typing import Optional, Tuple, Any, Union, Dict
from pathlib import Path

try:
    import librosa
    import soundfile as sf
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    MISSING_DEPS = str(e)

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # If running as main script, try absolute import
    from video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioSpectrogramCapture(VideoCaptureBase):
    """
    Capture audio spectrograms as video frames from microphones or audio files.
    Treats spectrograms as visual data that can be processed like regular video frames.
    """
    
    def __init__(self, source: Union[int, str, None] = None, **kwargs):
        """
        Initialize audio spectrogram capture.
        
        Args:
            source: Audio source - microphone index (int), file path (str), or None for default mic
            **kwargs: Spectrogram parameters:
                - n_mels: Number of mel bands (default: 128)
                - n_fft: FFT window size (default: 2048)
                - hop_length: Number of samples between successive frames (default: 512)
                - window_duration: Duration of audio window in seconds (default: 2.0)
                - freq_range: Frequency range tuple (min_freq, max_freq) (default: (20, 8000))
                - sample_rate: Audio sample rate (default: 44100)
                - colormap: OpenCV colormap for visualization (default: None for grayscale)
                - db_range: Dynamic range in dB (default: (-80, 0))
                - frame_rate: Spectrogram update rate in Hz (default: 30)
                - audio_buffer_size: Audio buffer size in samples (default: 1024)
        """
        if not AUDIO_AVAILABLE:
            raise ImportError(f"Audio dependencies not available: {MISSING_DEPS}. "
                            "Install with: pip install librosa soundfile pyaudio")
        
        super().__init__(source, **kwargs)
        
        # Spectrogram parameters
        self.n_mels = int(kwargs.get('n_mels', 128))
        self.n_fft = int(kwargs.get('n_fft', 2048))
        self.hop_length = int(kwargs.get('hop_length', 512))
        self.window_duration = float(kwargs.get('window_duration', 2.0))
        self.freq_range = tuple(map(int, kwargs.get('freq_range', '20,8000').split(',')))
        self.sample_rate = int(kwargs.get('sample_rate', 44100))  # Increased to support higher frequencies
        self.colormap = kwargs.get('colormap', None)  # None means grayscale (no colormap applied)
        self.db_range = kwargs.get('db_range', (-80, 0))
        self.frame_rate = int(kwargs.get('frame_rate', 30))
        self.audio_buffer_size = int(kwargs.get('audio_buffer_size', 1024))

        # Contrast enhancement parameters
        self.contrast_method = kwargs.get('contrast_method', 'fixed')  # 'fixed', 'adaptive', 'percentile'
        self.adaptive_alpha = float(kwargs.get('adaptive_alpha', 0.95))  # Smoothing factor for adaptive normalization
        self.percentile_range = tuple(map(int, kwargs.get('percentile_range', (5, 95))))
        self.gamma_correction = float(kwargs.get('gamma_correction', 1.0))  # Gamma correction for contrast
        self.noise_floor = float(kwargs.get('noise_floor', -70))  # Noise floor in dB

        # Adaptive normalization state
        self._adaptive_min = None
        self._adaptive_max = None
        
        # Validate and adjust sample rate based on frequency range
        max_freq = self.freq_range[1]
        nyquist_freq = self.sample_rate / 2
        if max_freq > nyquist_freq:
            # Automatically adjust sample rate to support the requested frequency range
            required_sample_rate = int(max_freq * 2.2)  # Add 10% margin above Nyquist
            # Round to common sample rates
            common_rates = [22050, 44100, 48000, 88200, 96000, 192000]
            self.sample_rate = min(rate for rate in common_rates if rate >= required_sample_rate)
            logger.warning(f"Frequency range {self.freq_range} requires sample rate >= {required_sample_rate}Hz. "
                          f"Adjusted sample rate to {self.sample_rate}Hz")
        
        logger.info(f"Sample rate: {self.sample_rate}Hz, Nyquist frequency: {self.sample_rate/2}Hz")
        
        # Calculated parameters
        self.window_samples = int(self.window_duration * self.sample_rate)
        self.spectrogram_width = int(self.window_samples // self.hop_length) + 1
        
        # Audio processing
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.mel_filter: Optional[np.ndarray] = None
        self.pyaudio_instance = None
        self.audio_stream = None
        self.audio_thread = None
        self.audio_stop_event = None
        self.is_file_source = False
        self.audio_data: Optional[np.ndarray] = None
        self.audio_position = 0
        
        # Frame buffer for consistent frame rate
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Background thread variables
        self._capture_thread = None
        self._stop_event = None
        self._latest_frame = None
        
        logger.info(f"AudioSpectrogramCapture initialized - Source: {source}")
        logger.info(f"Spectrogram params: n_mels={self.n_mels}, n_fft={self.n_fft}, "
                   f"window_duration={self.window_duration}s, freq_range={self.freq_range}")
    
    def start_async(self):
        """
        Start background thread to continuously generate spectrogram frames.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return  # Already running
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._capture_thread = threading.Thread(target=self._background_capture, daemon=True)
        self._capture_thread.start()
        logger.info("Started background spectrogram capture thread")

    def stop(self):
        """
        Stop background spectrogram capture thread.
        """
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, '_capture_thread') and self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        self._capture_thread = None
        self._stop_event = None
        logger.info("Stopped background spectrogram capture thread")

    def _background_capture(self):
        """Background thread function for continuous spectrogram generation."""
        frame_interval = 1.0 / self.frame_rate
        while not self._stop_event.is_set():  # type: ignore
            start_time = time.time()
            success, frame = self._read_direct()
            if success:
                with self.frame_lock:
                    self._latest_frame = frame
            
            # Maintain consistent frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent spectrogram frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        with self.frame_lock:
            frame = getattr(self, '_latest_frame', None)
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a spectrogram frame (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected:
            return False, None
        
        try:
            if self.is_file_source:
                return self._read_file_frame()
            else:
                return self._read_microphone_frame()
                
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def connect(self) -> bool:
        """Connect to audio source."""
        try:
            self._setup_mel_filter()
            
            if isinstance(self.source, str) and Path(self.source).exists():
                return self._connect_file()
            else:
                return self._connect_microphone()
                
        except Exception as e:
            logger.error(f"Failed to connect to audio source: {e}")
            return False
    
    def _setup_mel_filter(self):
        """Setup mel filter bank."""
        self.mel_filter = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.freq_range[0],
            fmax=self.freq_range[1]
        )
        if self.mel_filter is not None:
            logger.info(f"Mel filter bank created: {self.mel_filter.shape}")
        else:
            logger.error("Failed to create mel filter bank")
    
    def _connect_file(self) -> bool:
        """Connect to audio file."""
        try:
            self.audio_data, sr = librosa.load(self.source, sr=self.sample_rate)
            if sr != self.sample_rate:
                logger.warning(f"File sample rate {sr} resampled to {self.sample_rate}")
            
            self.is_file_source = True
            self.audio_position = 0
            self.is_connected = True
            
            logger.info(f"Connected to audio file: {self.source} "
                       f"(duration: {len(self.audio_data)/self.sample_rate:.2f}s)" if self.audio_data is not None else "")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return False
    
    def _connect_microphone(self) -> bool:
        """Connect to microphone."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Determine device index
            device_index = None
            if isinstance(self.source, int):
                device_index = self.source
            
            # Get device info
            if device_index is not None:
                device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
                logger.info(f"Using audio device: {device_info['name']}")
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.audio_buffer_size,
                stream_callback=self._audio_callback
            )
            
            self.is_file_source = False
            self.is_connected = True
            
            logger.info(f"Connected to microphone (device {device_index})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to microphone: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio capture."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Shift buffer and add new data
        shift_amount = len(audio_chunk)
        self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
        self.audio_buffer[-shift_amount:] = audio_chunk
        
        return (None, pyaudio.paContinue)
    
    def disconnect(self) -> bool:
        """Disconnect from audio source."""
        try:
            # Stop background thread first
            self.stop()
            
            self.is_connected = False
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            logger.info("Disconnected from audio source")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False
    
    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame captured by the background thread, or fall back to direct read if not running.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()
    
    def _read_file_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from audio file."""
        if self.audio_data is None:
            return False, None
            
        if self.audio_position + self.window_samples > len(self.audio_data):
            # Loop back to beginning
            self.audio_position = 0
        
        # Extract audio window
        audio_window = self.audio_data[self.audio_position:self.audio_position + self.window_samples]
        
        # Advance position
        advance_samples = int(self.sample_rate / self.frame_rate)
        self.audio_position += advance_samples
        
        # Generate spectrogram
        spectrogram = self._generate_spectrogram(audio_window)
        return True, spectrogram
    
    def _read_microphone_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from microphone."""
        if not self.audio_stream or not self.audio_stream.is_active():
            return False, None
        
        # Use current audio buffer
        audio_window = self.audio_buffer.copy()
        
        # Generate spectrogram
        spectrogram = self._generate_spectrogram(audio_window)
        return True, spectrogram
    
    def _generate_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate mel spectrogram from audio data."""
        if self.mel_filter is None:
            raise RuntimeError("Mel filter not initialized")
            
        # Compute STFT
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Apply mel filter
        mel_spectrogram = np.dot(self.mel_filter, magnitude)
        
        # Convert to dB
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Apply noise floor
        mel_db = np.maximum(mel_db, self.noise_floor)
        
        # Apply contrast enhancement based on method
        if self.contrast_method == 'adaptive':
            mel_normalized = self._adaptive_normalize(mel_db)
        elif self.contrast_method == 'percentile':
            mel_normalized = self._percentile_normalize(mel_db)
        else:  # 'fixed'
            mel_normalized = self._fixed_normalize(mel_db)
        
        # Apply gamma correction for additional contrast control
        if self.gamma_correction != 1.0:
            mel_normalized = np.power(mel_normalized, self.gamma_correction)
        
        # Convert to uint8
        mel_uint8 = (np.clip(mel_normalized, 0, 1) * 255).astype(np.uint8)
        
        # Flip vertically (high frequencies at top)
        mel_uint8 = np.flipud(mel_uint8)
        
        # Apply colormap if specified, otherwise return grayscale
        if self.colormap is not None:
            colored = cv2.applyColorMap(mel_uint8, self.colormap)
        else:
            # Convert grayscale to 3-channel for consistency with colored spectrograms
            colored = cv2.cvtColor(mel_uint8, cv2.COLOR_GRAY2BGR)
        
        return colored
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get spectrogram frame dimensions (width, height)."""
        return (self.spectrogram_width, self.n_mels)
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set spectrogram dimensions by adjusting parameters."""
        logger.warning("Frame size for spectrograms is determined by audio parameters. "
                      "Adjust n_mels, window_duration, or hop_length instead.")
        return False
    
    def get_fps(self) -> Optional[float]:
        """Get spectrogram frame rate."""
        return self.frame_rate
    
    def set_fps(self, fps: float) -> bool:
        """Set spectrogram frame rate."""
        if fps > 0:
            self.frame_rate = fps
            return True
        return False
    
    # Audio-specific parameter methods
    def get_n_mels(self) -> int:
        """Get number of mel bands."""
        return self.n_mels
    
    def set_n_mels(self, n_mels: int) -> bool:
        """Set number of mel bands (requires reconnection)."""
        if n_mels > 0:
            self.n_mels = n_mels
            logger.info(f"n_mels set to {n_mels}. Reconnect to apply changes.")
            return True
        return False
    
    def get_window_duration(self) -> float:
        """Get audio window duration."""
        return self.window_duration
    
    def set_window_duration(self, duration: float) -> bool:
        """Set audio window duration (requires reconnection)."""
        if duration > 0:
            self.window_duration = duration
            self.window_samples = int(duration * self.sample_rate)
            self.spectrogram_width = int(self.window_samples // self.hop_length) + 1
            logger.info(f"Window duration set to {duration}s. Reconnect to apply changes.")
            return True
        return False
    
    def get_freq_range(self) -> Tuple[float, float]:
        """Get frequency range."""
        return self.freq_range
    
    def set_freq_range(self, min_freq: float, max_freq: float) -> bool:
        """Set frequency range (requires reconnection). Automatically adjusts sample rate if needed."""
        if 0 < min_freq < max_freq:
            # Check if current sample rate supports the requested frequency range
            nyquist_freq = self.sample_rate / 2
            if max_freq > nyquist_freq:
                # Automatically adjust sample rate to support the requested frequency range
                required_sample_rate = int(max_freq * 2.2)  # Add 10% margin above Nyquist
                # Round to common sample rates
                common_rates = [22050, 44100, 48000, 88200, 96000, 192000]
                new_sample_rate = min(rate for rate in common_rates if rate >= required_sample_rate)
                
                logger.warning(f"Frequency range ({min_freq}, {max_freq}) requires sample rate >= {required_sample_rate}Hz. "
                              f"Adjusted sample rate from {self.sample_rate}Hz to {new_sample_rate}Hz")
                self.sample_rate = new_sample_rate
                
                # Recalculate window samples based on new sample rate
                self.window_samples = int(self.window_duration * self.sample_rate)
                self.spectrogram_width = int(self.window_samples // self.hop_length) + 1
            
            self.freq_range = (min_freq, max_freq)
            logger.info(f"Frequency range set to {self.freq_range}. Sample rate: {self.sample_rate}Hz. Reconnect to apply changes.")
            return True
        return False
    
    def get_nyquist_frequency(self) -> float:
        """Get the Nyquist frequency (maximum representable frequency)."""
        return self.sample_rate / 2.0
    
    def get_sample_rate(self) -> int:
        """Get the current sample rate."""
        return self.sample_rate
    
    def set_sample_rate(self, sample_rate: int) -> bool:
        """Set sample rate (requires reconnection)."""
        if sample_rate > 0:
            self.sample_rate = sample_rate
            # Recalculate dependent parameters
            self.window_samples = int(self.window_duration * self.sample_rate)
            self.spectrogram_width = int(self.window_samples // self.hop_length) + 1
            logger.info(f"Sample rate set to {sample_rate}Hz. Nyquist frequency: {self.get_nyquist_frequency()}Hz. Reconnect to apply changes.")
            return True
        return False

    def set_colormap(self, colormap: Optional[int]) -> bool:
        """
        Set the colormap for spectrogram visualization.
        
        Args:
            colormap: OpenCV colormap constant (e.g., cv2.COLORMAP_VIRIDIS) or None for grayscale
            
        Returns:
            bool: True if successful
        """
        self.colormap = colormap
        colormap_name = "grayscale" if colormap is None else f"cv2 colormap {colormap}"
        logger.info(f"Colormap set to {colormap_name}")
        return True
    
    def get_colormap(self) -> Optional[int]:
        """Get the current colormap (None means grayscale)."""
        return self.colormap

    def set_contrast_method(self, method: str) -> bool:
        """
        Set the contrast enhancement method.
        
        Args:
            method: 'fixed', 'adaptive', or 'percentile'
            
        Returns:
            bool: True if successful
        """
        if method in ['fixed', 'adaptive', 'percentile']:
            self.contrast_method = method
            # Reset adaptive state when changing methods
            if method == 'adaptive':
                self._adaptive_min = None
                self._adaptive_max = None
            logger.info(f"Contrast method set to {method}")
            return True
        else:
            logger.warning(f"Invalid contrast method: {method}. Use 'fixed', 'adaptive', or 'percentile'")
            return False
    
    def get_contrast_method(self) -> str:
        """Get the current contrast enhancement method."""
        return self.contrast_method
    
    def set_gamma_correction(self, gamma: float) -> bool:
        """
        Set gamma correction value for contrast enhancement.
        
        Args:
            gamma: Gamma value (< 1.0 increases contrast, > 1.0 decreases contrast)
            
        Returns:
            bool: True if successful
        """
        if gamma > 0:
            self.gamma_correction = gamma
            logger.info(f"Gamma correction set to {gamma}")
            return True
        return False
    
    def get_gamma_correction(self) -> float:
        """Get the current gamma correction value."""
        return self.gamma_correction
    
    def set_noise_floor(self, noise_floor_db: float) -> bool:
        """
        Set the noise floor in dB to suppress background noise.
        
        Args:
            noise_floor_db: Noise floor in dB (e.g., -70)
            
        Returns:
            bool: True if successful
        """
        self.noise_floor = noise_floor_db
        logger.info(f"Noise floor set to {noise_floor_db} dB")
        return True
    
    def get_noise_floor(self) -> float:
        """Get the current noise floor in dB."""
        return self.noise_floor
    
    def set_percentile_range(self, low: float, high: float) -> bool:
        """
        Set the percentile range for percentile-based normalization.
        
        Args:
            low: Lower percentile (0-100)
            high: Upper percentile (0-100)
            
        Returns:
            bool: True if successful
        """
        if 0 <= low < high <= 100:
            self.percentile_range = (low, high)
            logger.info(f"Percentile range set to {self.percentile_range}")
            return True
        return False
    
    def get_percentile_range(self) -> Tuple[float, float]:
        """Get the current percentile range."""
        return self.percentile_range

    def validate_frequency_range(self, min_freq: float, max_freq: float) -> Tuple[bool, str]:
        """
        Validate if the frequency range is supported by the current configuration.
        
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        nyquist = self.sample_rate / 2.0
        
        if min_freq <= 0:
            return False, f"Minimum frequency must be > 0, got {min_freq}"
        if max_freq <= min_freq:
            return False, f"Maximum frequency must be > minimum frequency, got {max_freq} <= {min_freq}"
        if max_freq > nyquist:
            return False, f"Maximum frequency {max_freq}Hz exceeds Nyquist limit {nyquist}Hz for sample rate {self.sample_rate}Hz"
        
        return True, "Frequency range is valid"

    # Stub methods for base class compatibility
    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """Not applicable for audio capture."""
        return True
    
    def set_exposure(self, value: float) -> bool:
        """Not applicable for audio capture."""
        return True
    
    def get_exposure(self) -> Optional[float]:
        """Not applicable for audio capture."""
        return None
    
    def set_gain(self, value: float) -> bool:
        """Audio gain control could be implemented here."""
        return True
    
    def get_gain(self) -> Optional[float]:
        """Audio gain control could be implemented here."""
        return None
    
    def _fixed_normalize(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Apply fixed dB range normalization.
        
        Args:
            mel_db: Mel spectrogram in dB
            
        Returns:
            np.ndarray: Normalized spectrogram [0, 1]
        """
        return np.clip((mel_db - self.db_range[0]) / (self.db_range[1] - self.db_range[0]), 0, 1)
    
    def _adaptive_normalize(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Apply adaptive normalization based on current frame statistics.
        
        Args:
            mel_db: Mel spectrogram in dB
            
        Returns:
            np.ndarray: Normalized spectrogram [0, 1]
        """
        # Initialize adaptive min/max values
        if self._adaptive_min is None or self._adaptive_max is None:
            self._adaptive_min = np.min(mel_db)
            self._adaptive_max = np.max(mel_db)
        
        # Update adaptive min/max values using exponential moving average
        current_min = np.min(mel_db)
        current_max = np.max(mel_db)
        
        self._adaptive_min = self.adaptive_alpha * self._adaptive_min + (1 - self.adaptive_alpha) * current_min
        self._adaptive_max = self.adaptive_alpha * self._adaptive_max + (1 - self.adaptive_alpha) * current_max
        
        # Ensure we don't divide by zero
        range_val = max(self._adaptive_max - self._adaptive_min, 1e-10)
        
        # Normalize
        normalized = (mel_db - self._adaptive_min) / range_val
        return np.clip(normalized, 0, 1)
    
    def _percentile_normalize(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Apply percentile-based normalization for robust contrast.
        
        Args:
            mel_db: Mel spectrogram in dB
            
        Returns:
            np.ndarray: Normalized spectrogram [0, 1]
        """
        # Calculate percentiles
        low_percentile = np.percentile(mel_db, self.percentile_range[0])
        high_percentile = np.percentile(mel_db, self.percentile_range[1])
        
        # Ensure we don't divide by zero
        range_val = max(high_percentile - low_percentile, 1e-10)
        
        # Normalize
        normalized = (mel_db - low_percentile) / range_val
        return np.clip(normalized, 0, 1)

    @classmethod
    def discover(cls) -> list:
        """
        Discover available audio input devices (microphones).
        
        Returns:
            list: List of dictionaries containing audio device information.
                Each dict contains: {'index': int, 'name': str, 'channels': int, 'sample_rate': float}
        """
        if not AUDIO_AVAILABLE:
            logger.warning("Audio dependencies not available. Cannot discover audio devices.")
            return []
        
        devices = []
        
        try:
            p = pyaudio.PyAudio()
            
            # Get device count
            device_count = p.get_device_count()
            
            for i in range(device_count):
                try:
                    device_info = p.get_device_info_by_index(i)
                    
                    # Only include input devices (microphones)
                    max_input_channels = device_info.get('maxInputChannels', 0)
                    if isinstance(max_input_channels, (int, float)) and max_input_channels > 0:
                        device_data = {
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': device_info['defaultSampleRate']
                        }
                        devices.append(device_data)
                        logger.info(f"Found audio input device: {device_data}")
                        
                except Exception as e:
                    logger.warning(f"Could not get info for audio device {i}: {e}")
                    continue
            
            p.terminate()
            
        except Exception as e:
            logger.error(f"Error discovering audio devices: {e}")
        
        return devices

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for audio spectrogram capture"""
        return {
            'title': 'Audio Spectrogram Configuration',
            'description': 'Configure audio spectrogram capture from microphones or audio files',
            'fields': [
                {
                    'name': 'source',
                    'label': 'Audio Source',
                    'type': 'text',
                    'placeholder': '0 or /path/to/audio.wav',
                    'description': 'Microphone index (0, 1, 2...) or path to audio file',
                    'required': False,
                    'default': 0
                },
                {
                    'name': 'n_mels',
                    'label': 'Mel Bands',
                    'type': 'number',
                    'min': 32,
                    'max': 256,
                    'placeholder': '128',
                    'description': 'Number of mel frequency bands in spectrogram',
                    'required': False,
                    'default': 128
                },
                {
                    'name': 'n_fft',
                    'label': 'FFT Window Size',
                    'type': 'select',
                    'options': [
                        {'value': 512, 'label': '512'},
                        {'value': 1024, 'label': '1024'},
                        {'value': 2048, 'label': '2048'},
                        {'value': 4096, 'label': '4096'}
                    ],
                    'description': 'FFT window size for frequency analysis',
                    'required': False,
                    'default': 2048
                },
                {
                    'name': 'hop_length',
                    'label': 'Hop Length',
                    'type': 'number',
                    'min': 128,
                    'max': 2048,
                    'placeholder': '512',
                    'description': 'Number of samples between successive frames',
                    'required': False,
                    'default': 512
                },
                {
                    'name': 'window_duration',
                    'label': 'Window Duration (s)',
                    'type': 'number',
                    'min': 0.5,
                    'max': 10.0,
                    'step': 0.1,
                    'placeholder': '2.0',
                    'description': 'Duration of audio window in seconds',
                    'required': False,
                    'default': 2.0
                },
                {
                    'name': 'sample_rate',
                    'label': 'Sample Rate (Hz)',
                    'type': 'select',
                    'options': [
                        {'value': 22050, 'label': '22050'},
                        {'value': 44100, 'label': '44100'},
                        {'value': 48000, 'label': '48000'},
                        {'value': 96000, 'label': '96000'}
                    ],
                    'description': 'Audio sample rate in Hz',
                    'required': False,
                    'default': 44100
                },
                {
                    'name': 'freq_range',
                    'label': 'Frequency Range (Hz)',
                    'type': 'text',
                    'placeholder': '20,8000',
                    'description': 'Frequency range as "min,max" (e.g., "20,8000")',
                    'required': False,
                    'default': '20,8000'
                },
                {
                    'name': 'frame_rate',
                    'label': 'Frame Rate (FPS)',
                    'type': 'number',
                    'min': 1,
                    'max': 60,
                    'placeholder': '30',
                    'description': 'Spectrogram update rate in frames per second',
                    'required': False,
                    'default': 30
                },
                {
                    'name': 'colormap',
                    'label': 'Color Map',
                    'type': 'select',
                    'options': [
                        {'value': None, 'label': 'Grayscale'},
                        {'value': 2, 'label': 'Jet'},
                        {'value': 9, 'label': 'Hot'},
                        {'value': 11, 'label': 'Viridis'},
                        {'value': 13, 'label': 'Plasma'},
                        {'value': 21, 'label': 'Turbo'}
                    ],
                    'description': 'Color mapping for spectrogram visualization',
                    'required': False,
                    'default': None
                },
                {
                    'name': 'contrast_method',
                    'label': 'Contrast Method',
                    'type': 'select',
                    'options': [
                        {'value': 'fixed', 'label': 'Fixed Range'},
                        {'value': 'adaptive', 'label': 'Adaptive'},
                        {'value': 'percentile', 'label': 'Percentile'}
                    ],
                    'description': 'Method for contrast enhancement',
                    'required': False,
                    'default': 'fixed'
                },
                {
                    'name': 'gamma_correction',
                    'label': 'Gamma Correction',
                    'type': 'number',
                    'min': 0.1,
                    'max': 3.0,
                    'step': 0.1,
                    'placeholder': '1.0',
                    'description': 'Gamma correction for contrast (< 1.0 increases contrast)',
                    'required': False,
                    'default': 1.0
                },
                {
                    'name': 'noise_floor',
                    'label': 'Noise Floor (dB)',
                    'type': 'number',
                    'min': -100,
                    'max': -20,
                    'placeholder': '-70',
                    'description': 'Noise floor in dB to suppress background noise',
                    'required': False,
                    'default': -70
                }
            ]
        }


if __name__ == "__main__":
    import queue
    import threading
    import time
    import cv2

    # Example usage: Capture spectrograms from multiple audio sources (microphones or files)
    
    devices = AudioSpectrogramCapture.discover()
    print(f"Discovered {len(devices)} audio input devices:")
    for device in devices:
       print(f" - {device['name']} (Index: {device['index']}, Channels: {device['channels']}, Sample Rate: {device['sample_rate']})")
