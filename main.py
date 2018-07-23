import glob
import os
import librosa
import numpy as np
from joblib import Parallel, delayed
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import  preprocessing


def get_class(path):
    return path.split('audio/')[1].split('_0')[0].split('_t')[0]


def append_path(path):
    return './data/train/audio/'+path


sample_rate = 16000
num_total_classes = 10
n_bands = 75
n_frames = 75
meta = pd.read_csv('data/train/meta.txt', delim_whitespace=True)
train_paths = np.vectorize(append_path)(meta.values[:,0])
train_labels = meta.values[:,4]
test_paths = np.sort(np.array(glob.glob('./data/test/audio/*.wav')))
test_labels = np.vectorize(get_class)(test_paths)
le = preprocessing.LabelEncoder()
test_labels = le.fit_transform(test_labels)
train_labels = le.transform(train_labels)


def read_audio(audio_path, target_fs=None):
    (audio, fs) = librosa.load(audio_path, sr=None)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def event_detector(x, duration):
    event_beggining = 0
    x_new = x
    if x.shape[0] > duration:
        x_sum = np.mean(x, axis=1)
        max_sum = np.sum(x_sum[:duration])
        prev_sum = max_sum
        for i in range(x_sum.shape[0]-duration):
            new_sum = prev_sum - x_sum[i]+x_sum[i+duration]
            if new_sum > max_sum:
                max_sum = new_sum
                event_beggining = i+1
            prev_sum = new_sum
        x_new = x[event_beggining:event_beggining+duration,:]
    elif x.shape[0] < duration:
        pad_shape = (duration-x.shape[0],x.shape[1])
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad))
    return x_new


def extract_features(paths, bands, frames):
    n_window = int(sample_rate * 1. / frames * 2)
    n_overlap = int(n_window/2.)
    melW = librosa.filters.mel(sr=sample_rate, n_fft=n_window, n_mels=bands, fmin=0., fmax=8000.)
    ham_win = np.hamming(n_window)
    log_specgrams_list = []
    for waw_path in paths:
        print(waw_path)
        sound_clip, fn_fs = read_audio(waw_path, target_fs=sample_rate)
        assert (int(fn_fs) == sample_rate)
        if sound_clip.shape[0] == 0:
            print("File %s is corrupted!" % waw_path)
            continue
        [f, t, x] = signal.spectral.spectrogram(
                x=sound_clip,
                window=ham_win,
                nperseg=n_window,
                noverlap=n_overlap,
                detrend=False,
                return_onesided=True,
                mode='magnitude')
        x = event_detector(x.T, frames)
        x = np.dot(x, melW.T)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32).T
        log_specgrams_list.append(x)
    log_specgrams = np.asarray(log_specgrams_list).reshape(len(log_specgrams_list), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    features = np.concatenate((features, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 1])
    # librosa.display.specshow(features[0,:,:,0], sr=sample_rate, x_axis='time', y_axis='mel',
    #                          x_coords=np.linspace(0, 1, features[0,:,:,1].shape[1]))
    # plt.xlabel("Time (s)")
    # plt.show()
    # librosa.display.specshow(features[0,:,:,1], sr=sample_rate, x_axis='time', y_axis='mel',
    #                          x_coords=np.linspace(0, 1, features[0,:,:,1].shape[1]))
    # plt.xlabel("Time (s)")
    # plt.show()
    # librosa.display.specshow(features[0,:,:,2], sr=sample_rate, x_axis='time', y_axis='mel',
    #                          x_coords=np.linspace(0, 1, features[0,:,:,1].shape[1]))
    # plt.xlabel("Time (s)")
    # plt.show()
    return features


train_features = extract_features(train_paths,n_bands,n_frames)
print('Train features has extracted')
print(train_features.shape)
np.save('train_features.npy',train_features)
np.save('train_labels.npy', train_labels)
