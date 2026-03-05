import librosa
import numpy as np
import joblib

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

model = joblib.load("cardioscope_model.pkl")

def extract_features(file_path):

    signal, sr = librosa.load(file_path, sr=None)

    signal = signal / np.max(np.abs(signal))

    lowcut = 20
    highcut = 400
    order = 4

    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    N = len(filtered)

    fft_values = fft(filtered)
    freq = fftfreq(N, 1/sr)

    positive_freq = freq[:N//2]
    magnitude = np.abs(fft_values[:N//2])

    normal_band = (positive_freq >= 20) & (positive_freq <= 150)
    murmur_band = (positive_freq >= 200) & (positive_freq <= 500)

    normal_energy = np.sum(magnitude[normal_band]**2)
    murmur_energy = np.sum(magnitude[murmur_band]**2)

    spectral_centroid = np.sum(positive_freq * magnitude) / (np.sum(magnitude) + 1e-8)

    mfcc = librosa.feature.mfcc(y=filtered, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    mean_amp = np.mean(filtered)
    std_amp = np.std(filtered)

    features = np.hstack([
        mean_amp,
        std_amp,
        spectral_centroid,
        normal_energy,
        murmur_energy,
        mfcc_mean
    ])

    return features

file_path = input("Enter heart sound file path: ")

features = extract_features(file_path)

features = features.reshape(1, -1)

prediction = model.predict(features)[0]

prob = model.predict_proba(features)[0]

confidence = np.max(prob)

print("\nPrediction:", prediction)
print("Confidence:", round(confidence,3))
