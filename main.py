import os
import librosa
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

base_folder = "training"

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

X = []
y = []

for folder in os.listdir(base_folder):

    folder_path = os.path.join(base_folder, folder)

    if os.path.isdir(folder_path):

        ref_path = os.path.join(folder_path, "REFERENCE.csv")

        if os.path.exists(ref_path):

            df = pd.read_csv(ref_path, header=None)

            for index, row in df.iterrows():

                file_name = row[0]+ ".wav"
                label = row[1]

                wav_path = os.path.join(folder_path, file_name)

                if os.path.exists(wav_path):

                    features = extract_features(wav_path)

                    X.append(features)
                    y.append(label)

X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))
print("Total samples loaded:", len(X))