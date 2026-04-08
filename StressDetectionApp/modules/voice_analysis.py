import librosa
import numpy as np
import joblib

# load trained voice model
voice_model = joblib.load("models/voice_model.joblib")


def extract_voice_features(audio_file):

    audio, sr = librosa.load(audio_file)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


def detect_voice_stress(audio_file):

    features = extract_voice_features(audio_file)

    features = features.reshape(1, -1)

    prediction = voice_model.predict(features)

    if prediction[0] == 0:
        return 0   # low

    elif prediction[0] == 1:
        return 1   # medium

    else:
        return 2   # high