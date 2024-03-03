
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile

from scipy import signal

rate = 100 # Hz
window_size = 256 # 0.5s
FFT_length = 256 # Length of the FFT used, if a zero padded FFT is desired. In this case, 12 zero padding is used
#data
def spectrogram(data, verbose=False):
    data1d = data.reshape(-1)
    if verbose:
        print("data1d.shape", data1d.shape)
    freqs, times, specs = signal.spectrogram(
        data1d,
        fs=rate, # Frequence Hz
        window="boxcar", # Rectangular segments
        nperseg=window_size, # 0.5s
        nfft=FFT_length, # Length of the FFT used, if a zero padded FFT is desired. In this case, 12 zero padding is used
        noverlap=0, # no overlap
        detrend=False,
        mode = 'magnitude') # retrieve complex number

    if verbose:
        print("freqs.shape", freqs.shape)
        print("specs.shape", specs.shape)

    return freqs, specs

def stft(data, verbose=False):
    data1d = data.reshape(-1)
    if verbose:
        print("data1d.shape", data1d.shape)
    freqs, times, specs = signal.stft(
        data1d,
        fs=rate, # Frequence Hz
        window="boxcar", # Rectangular segments
        nperseg=window_size, # 0.5s
        nfft=FFT_length, # Length of the FFT used, if a zero padded FFT is desired
        noverlap=0, # no overlap
        detrend=False,
        scaling = 'spectrum', # retrieve complex number
        boundary=None,
    ) 

    if verbose:
        print("freqs.shape", freqs.shape)
        print("specs.shape", specs.shape)

    return freqs, np.abs(specs)


