import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt

def frame_audio(audio_file_path, frame_size_sec, overlap_size_sec, window_type):
    # Read audio file
    sample_rate, signal = wavfile.read(audio_file_path)
    plt.plot(signal)
    plt.show()
    # Convert frame size and overlap size from seconds to samples
    frame_size = int(sample_rate * frame_size_sec)
    overlap_size = int(sample_rate * overlap_size_sec)

    # Compute the step size and number of frames
    step_size = frame_size - overlap_size
    num_frames = (len(signal) - overlap_size) // step_size

    # Initialize the frames matrix and window function
    frames = np.zeros((num_frames, frame_size))
    window = get_window(window_type, frame_size)

    # Fill the frames matrix with the signal values
    for i in range(num_frames):
        start_index = i * step_size
        end_index = start_index + frame_size
        frames[i] = signal[start_index:end_index] * window

    return frames


def compute_energy(frames):
    energy_vector = np.sum(frames ** 2, axis=1)

    return energy_vector


def compute_zero_crossing(frames):
    zero_crossing_vector = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2

    return zero_crossing_vector



audio_file_path = "C:\digitRec\wav\l_02.wav"
frame_size_sec = 0.02
overlap_size_sec = 0.01
window_type = 'hamming'

frames = frame_audio(audio_file_path, frame_size_sec, overlap_size_sec, window_type)
energy_vector = compute_energy(frames)

# Compute energy and zero crossing values
energy_vector = compute_energy(frames)
zero_crossing_vector = compute_zero_crossing(frames)

# Plot framed signal
framed_signal = frames.flatten()
plt.subplot(3, 1, 1)
plt.plot(framed_signal)
plt.title('Framed Signal')

# Plot energy vector
plt.subplot(3, 1, 2)
plt.plot(energy_vector)
plt.title('Energy Vector')

# Plot zero crossing vector
plt.subplot(3, 1, 3)
plt.plot(zero_crossing_vector)
plt.title('Zero Crossing Vector')

# Show plots
plt.show()