import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time
import struct
from scipy.io.wavfile import read



def get_mel_spectrogram(file_path, max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
       
        y, sr = librosa.load(file_path)

        
        normalized_y = librosa.util.normalize(y)

        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        mel_db = librosa.amplitude_to_db(abs(mel))

        normalized_mel = librosa.util.normalize(mel_db)

        shape = normalized_mel.shape[1]
        if (max_padding > 0 & shape < max_padding):
            xDiff = max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        return None 
    return normalized_mel


def add_padding(features, max_padding=174):
    padded = []
    for i in range(len(features)):
        px = features[i]
        
        size = len(px[i])
        if (size < max_padding):
            xDiff = max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded



def load_split_distributions(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    return [data['test_split_idx'], data['train_split_idx']]

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    return train_score, test_score

def acc_per_class(np_probs_array):    
    accs = []
    for idx in range(0, np_probs_array.shape[0]):
        correct = np_probs_array[idx][idx].astype(int)
        total = np_probs_array[idx].sum().astype(int)
        acc = (correct / total) * 100
        accs.append(acc)
    return accs

def last_fun(new):
    with open("test_both_ml_and_db.py", 'r') as file:
            new_file = []
            for line in file:
                if "pth" in line:
                    new_file.append(line.split("pth")[0] + "pth = "+ '"' + new + '"')
                else:
                    new_file.append(line)
            with open("test_both_ml_and_db.py", 'w') as file:
                for line in new_file:
                    file.writelines(line)
            