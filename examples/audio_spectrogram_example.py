#!/usr/bin/env python3
"""
Audio Spectrogram Capture Example

Demonstrates audio spectrogram capture from microphone or audio file with advanced
contrast controls, colormap options, and real-time parameter adjustment.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test audio spectrogram capture with interactive controls."""
    cv2.namedWindow("Audio Spectrogram", cv2.WINDOW_NORMAL)
    print("Testing Audio Spectrogram Capture:")
    
    # Default audio parameters for good visualization
    audio_params = {
        'n_mels': 128,
        'n_fft': 2048,
        'window_duration': 3.0,
        'freq_range': (20, 8000),
        'frame_rate': 30,
        'colormap': cv2.COLORMAP_VIRIDIS,
        'contrast_method': 'adaptive',
        'gamma_correction': 1.0,
        'noise_floor': -60,
        'percentile_range': (5, 95)
    }
    
    frame_source = FrameSourceFactory.create('audio_spectrogram', source=None, **audio_params)
    
    if not frame_source.connect():
        print("Failed to connect to audio source")
        return
    
    frame_source.start_async()
    print("Started background spectrogram capture thread")
    
    if frame_source.is_connected:
        print(f"Audio spectrogram parameters:")
        print(f"  Frame size: {frame_source.get_frame_size()}")
        print(f"  FPS: {frame_source.get_fps()}")
        
        # Print audio-specific parameters
        if hasattr(frame_source, 'get_n_mels'):
            print(f"  N mels: {frame_source.get_n_mels()}")
        if hasattr(frame_source, 'get_window_duration'):
            print(f"  Window duration: {frame_source.get_window_duration()}s")
        if hasattr(frame_source, 'get_freq_range'):
            print(f"  Frequency range: {frame_source.get_freq_range()}")
        if hasattr(frame_source, 'get_sample_rate'):
            print(f"  Sample rate: {frame_source.get_sample_rate()}Hz")
        if hasattr(frame_source, 'get_contrast_method'):
            print(f"  Contrast method: {frame_source.get_contrast_method()}")
        if hasattr(frame_source, 'get_gamma_correction'):
            print(f"  Gamma correction: {frame_source.get_gamma_correction():.2f}")
        if hasattr(frame_source, 'get_noise_floor'):
            print(f"  Noise floor: {frame_source.get_noise_floor()} dB")
        if hasattr(frame_source, 'get_percentile_range'):
            print(f"  Percentile range: {frame_source.get_percentile_range()}%")
        
        def print_help():
            print("\nAudio Spectrogram Controls:")
            print("  ESC - Quit")
            print("  h - Show this help")
            print("  0 - Grayscale (default)")
            print("  1 - Viridis colormap")
            print("  2 - Plasma colormap")
            print("  3 - Inferno colormap")
            print("  4 - Hot colormap")
            print("  5 - Jet colormap")
            print("  +/- - Adjust mel bands (requires restart)")
            print("  c - Cycle contrast methods (fixed/adaptive/percentile)")
            print("  g/G - Decrease/increase gamma correction")
            print("  n/N - Decrease/increase noise floor")
            print("  p/P - Adjust percentile range")
        
        print_help()
        
        while frame_source.is_connected:
            ret, frame = frame_source.read()
            if ret and frame is not None:
                cv2.imshow("Audio Spectrogram", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit
                break
            elif key == ord('h'):  # Show help
                print_help()
            elif key == ord('0') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(None)
                print("Colormap: Grayscale")
            elif key == ord('1') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_VIRIDIS)
                print("Colormap: Viridis")
            elif key == ord('2') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_PLASMA)
                print("Colormap: Plasma")
            elif key == ord('3') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_INFERNO)
                print("Colormap: Inferno")
            elif key == ord('4') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_HOT)
                print("Colormap: Hot")
            elif key == ord('5') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_JET)
                print("Colormap: Jet")
            elif key == ord('+') or key == ord('='):
                if hasattr(frame_source, 'get_n_mels') and hasattr(frame_source, 'set_n_mels'):
                    current_mels = frame_source.get_n_mels()
                    frame_source.set_n_mels(min(current_mels + 16, 256))
                    print(f"Mel bands: {frame_source.get_n_mels()} (restart to apply)")
            elif key == ord('-'):
                if hasattr(frame_source, 'get_n_mels') and hasattr(frame_source, 'set_n_mels'):
                    current_mels = frame_source.get_n_mels()
                    frame_source.set_n_mels(max(current_mels - 16, 32))
                    print(f"Mel bands: {frame_source.get_n_mels()} (restart to apply)")
            elif key == ord('c'):  # Cycle contrast methods
                if hasattr(frame_source, 'get_contrast_method') and hasattr(frame_source, 'set_contrast_method'):
                    current_method = frame_source.get_contrast_method()
                    methods = ['fixed', 'adaptive', 'percentile']
                    current_index = methods.index(current_method)
                    next_method = methods[(current_index + 1) % len(methods)]
                    frame_source.set_contrast_method(next_method)
                    print(f"Contrast method: {next_method}")
            elif key == ord('g'):  # Decrease gamma
                if hasattr(frame_source, 'get_gamma_correction') and hasattr(frame_source, 'set_gamma_correction'):
                    current_gamma = frame_source.get_gamma_correction()
                    new_gamma = max(current_gamma - 0.1, 0.1)
                    frame_source.set_gamma_correction(new_gamma)
                    print(f"Gamma: {new_gamma:.2f} ({'more contrast' if new_gamma < 1.0 else 'less contrast'})")
            elif key == ord('G'):  # Increase gamma
                if hasattr(frame_source, 'get_gamma_correction') and hasattr(frame_source, 'set_gamma_correction'):
                    current_gamma = frame_source.get_gamma_correction()
                    new_gamma = min(current_gamma + 0.1, 3.0)
                    frame_source.set_gamma_correction(new_gamma)
                    print(f"Gamma: {new_gamma:.2f} ({'more contrast' if new_gamma < 1.0 else 'less contrast'})")
            elif key == ord('n'):  # Decrease noise floor
                if hasattr(frame_source, 'get_noise_floor') and hasattr(frame_source, 'set_noise_floor'):
                    current_floor = frame_source.get_noise_floor()
                    new_floor = max(current_floor - 5, -100)
                    frame_source.set_noise_floor(new_floor)
                    print(f"Noise floor: {new_floor} dB")
            elif key == ord('N'):  # Increase noise floor
                if hasattr(frame_source, 'get_noise_floor') and hasattr(frame_source, 'set_noise_floor'):
                    current_floor = frame_source.get_noise_floor()
                    new_floor = min(current_floor + 5, -10)
                    frame_source.set_noise_floor(new_floor)
                    print(f"Noise floor: {new_floor} dB")
            elif key == ord('p'):  # Decrease percentile range
                if hasattr(frame_source, 'get_percentile_range') and hasattr(frame_source, 'set_percentile_range'):
                    low, high = frame_source.get_percentile_range()
                    new_low = min(low + 2, 20)
                    new_high = max(high - 2, 80)
                    if new_low < new_high:
                        frame_source.set_percentile_range(new_low, new_high)
                        print(f"Percentile range: {new_low}-{new_high}% (more aggressive)")
            elif key == ord('P'):  # Increase percentile range
                if hasattr(frame_source, 'get_percentile_range') and hasattr(frame_source, 'set_percentile_range'):
                    low, high = frame_source.get_percentile_range()
                    new_low = max(low - 2, 0)
                    new_high = min(high + 2, 100)
                    frame_source.set_percentile_range(new_low, new_high)
                    print(f"Percentile range: {new_low}-{new_high}% (less aggressive)")
    
    frame_source.stop()
    frame_source.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
