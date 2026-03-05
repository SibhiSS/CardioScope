import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, filtfilt,hilbert,find_peaks
from scipy.fft import fft, fftfreq


signal, sr = librosa.load("data/a0001.wav", sr=None)
lowcut = 20
highcut = 400
order = 4

print("Sampling Rate:", sr)
print("Total Samples:", len(signal))

max_value = np.max(np.abs(signal))
print("Maximum amplitude:", max_value)
normalized_signal = signal / max_value
print("Normalized max amplitude:", np.max(np.abs(normalized_signal)))

time = np.arange(len(signal)) / sr

nyquist = 0.5 * sr
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(order, [low, high], btype='band')
filtered_signal = filtfilt(b, a, normalized_signal)
analytic_signal = hilbert(filtered_signal)
envelope = np.abs(analytic_signal)
N = len(filtered_signal)
fft_values = fft(filtered_signal)
freq = fftfreq(N, 1/sr)
positive_freq = freq[:N//2]
magnitude = np.abs(fft_values[:N//2])

#Detect Peaks
peaks, _ = find_peaks(envelope, distance=sr*0.5)


plt.figure(figsize=(12,4))
plt.plot(time, envelope)
plt.plot(time[peaks], envelope[peaks], "ro")
plt.title("Detected Heartbeat Peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

#estimating the heart rate
peak_times = time[peaks]

intervals = np.diff(peak_times)

average_interval = np.mean(intervals)

heart_rate = 60 / average_interval

print("Estimated Heart Rate:", heart_rate, "BPM")