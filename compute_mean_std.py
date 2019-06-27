"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
#import functools
import codecs
import random
from collections import Counter
import csv
import tensorflow
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc

def audioToLinearInputVector(samples, sample_rate, stride_size, window_size):

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape=nshape, strides=nstrides)
    assert np.all(
        windows[:, 1] == samples[stride_size:(stride_size + window_size)])
        # window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    return fft, freqs


def count_csv(counter, csv_data):
    count = 0
    for line in csv_data:
        count += 1
        for char in line[2]:
            counter.update(char)
        if count % 10000 == 0:
            print (count)


def main():
    count_threshold = 0
    with open("data/train_clean.csv", "rt", encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        count = 0
        csv_data = []
        for row in spamreader:
            csv_data.append(row)
            count += 1
            if count % 100000 == 0:
                print (count)
                #break
            #print (', '.join(row))    
    print (len(csv_data))
    csv_data = csv_data[1:]
    random.shuffle(csv_data)
    print (csv_data[:10])
    csv_data = csv_data[:10000]
    max_freq = 16000
    eps = 1e-14
    features = []
    for line in csv_data:
        print(line[0])
        fs, audio = wav.read(line[0])
        print (fs) #16000
        #specgram, freqs = audioToLinearInputVector(audio, fs, 160, 320) 
        #ind = np.where(freqs <= max_freq)[0][-1] + 1
        #linear = np.log(specgram[:ind, :] + eps)
        linear =  mfcc(audio, samplerate=fs, numcep=13, winlen=0.032, winstep=0.02, winfunc=np.hamming)
        print (linear.shape) # (161,x)
        features.append(linear)
    features = np.vstack(features)
    print (features.shape)
    mean = np.mean(features, axis=0).reshape([1, -1])
    std = np.std(features, axis=0).reshape([1, -1])
    np.savez('data/mean_std_train_clean.npz', mean=mean, std=std)


if __name__ == '__main__':
    main()
