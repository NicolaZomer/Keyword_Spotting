# preprocessing functions
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from pywt import dwt

import os
import glob
import importlib

from python_speech_features import logfbank, mfcc, delta

commands = [
  'backward',
  'bed',
  'bird',
  'cat',
  'dog',
  'down',
  'eight',
  'five',
  'follow',
  'forward',
  'four',
  'go',
  'happy',
  'house',
  'learn',
  'left',
  'marvin',
  'nine',
  'no',
  'off',
  'on',
  'one',
  'right',
  'seven',
  'sheila',
  'six',
  'stop',
  'three',
  'tree',
  'two',
  'up',
  'visual',
  'wow',
  'yes',
  'zero'
]

label_to_class = {commands[i]:i for i in range(len(commands))}
class_to_label = {i:commands[i] for i in range(len(commands))}

data_path='data/full_speech_commands'


def load_data(file_name, file_label, data_path_=data_path):
    if isinstance(file_name, bytes):
        file_name = file_name.decode()
    if isinstance(file_label, bytes):
        file_label = file_label.decode()
    if isinstance(data_path_, bytes):
        data_path_ = data_path_.decode()

    if not isinstance(file_label, str):
        file_label = class_to_label[file_label]

    file_path = data_path_ + '/' + file_label + '/' + file_name
    _, data = wavfile.read(file_path)

    return data.squeeze()


def padding_trimming(data, output_sequence_length=16000):
    data_shape = data.shape[0]
    
    # trimming
    if data_shape>output_sequence_length:
        data=data[:output_sequence_length]
    
    # padding
    elif data_shape<output_sequence_length:
        tot_pad = output_sequence_length-data_shape
        pad_before = int(np.ceil(tot_pad/2))
        pad_after  = int(np.floor(tot_pad/2))
        data = np.pad(data, pad_width=(pad_before, pad_after), mode='mean')
        
    return data


def background_noise(data, noise_dict, select_noise=None, noise_reduction=0.5):
    '''
    data: input audio signal, already loaded and preprocessed, it must be a numpy array 
    select_noise: decide what kind of noise to add to the input signal, by default a random choice 
    noise_reduction: set it to a value between 0 and 1 to reduce the amount of noise, by default 0.8
    '''
    
    target_size = data.shape[0]
    
    # None case
    if select_noise is None:
        select_noise = np.random.choice(np.arange(1, 7))
        
    # other cases, plus random selection  
    noise_data = noise_dict[str(select_noise)]
    N = noise_data.shape[0]
    
    # random cropping
    from_ = np.random.randint(0, int(N-target_size))
    to_ = from_ + target_size
    noise_data = noise_data[from_:to_]
    
    # add noise to input audio
    data_with_noise = data + (1-noise_reduction)*noise_data
    
    return data_with_noise


def get_spectrogram(
                    signal,                               # audio signal from which to compute features (N*1 array)
                    samplerate = 16000,                   # samplerate of the signal we are working with
                    winlen     = 25,                      # length of the analysis window (milliseconds)
                    winstep    = 10,                      # step between successive windows (milliseconds)
                    nfft       = 512,                     # FFT size
                    winfunc    = tf.signal.hamming_window # analysis window to apply to each frame
                    ):

    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tf.signal.stft(
                                signal.astype(float),
                                int(samplerate*winlen/1000),
                                int(samplerate*winstep/1000),
                                nfft,
                                winfunc
                                )

    # Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # Convert to NumPy array
    spectrogram = np.array(spectrogram)

    # Convert the frequencies to log scale and transpose, so that the time is represented on the x-axis (columns)
    # Add an epsilon to avoid taking a log of zero
    spectrogram = np.log(spectrogram.T + np.finfo(float).eps)

    return spectrogram


def get_logfbank(
                signal,             # audio signal from which to compute features (N*1 array)
                samplerate = 16000, # samplerate of the signal we are working with
                winlen     = 25,    # length of the analysis window (milliseconds)
                winstep    = 10,    # step between successive windows (milliseconds)
                nfilt      = 40,    # number of filters in the filterbank
                nfft       = 512,   # FFT size
                lowfreq    = 300,   # lowest band edge of mel filters (Hz)
                highfreq   = None,  # highest band edge of mel filters (Hz)
                ):

    if highfreq is None:
        highfreq = samplerate / 2

    # Extract log Mel-filterbank energy features
    logfbank_feat = logfbank(
                            signal,
                            samplerate,
                            winlen/1000,
                            winstep/1000,
                            nfilt,
                            nfft,
                            lowfreq,
                            highfreq,
                            )
    logfbank_feat = logfbank_feat.T

    return logfbank_feat


def get_mfcc(
            signal,                    # audio signal from which to compute features (N*1 array)
            samplerate   = 16000,      # samplerate of the signal we are working with
            winlen       = 25,         # length of the analysis window (milliseconds)
            winstep      = 10,         # step between successive windows (milliseconds)
            numcep       = 13,         # number of cepstrum to return
            nfilt        = 40,         # number of filters in the filterbank
            nfft         = 512,        # FFT size
            lowfreq      = 300,        # lowest band edge of mel filters (Hz)
            highfreq     = None,       # highest band edge of mel filters (Hz)
            appendEnergy = True,       # if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy
            winfunc      = np.hamming, # analysis window to apply to each frame
            delta_order  = 2,          # maximum order of the Delta features
            delta_window = 1           # window size for the Delta features
            ):

    if highfreq is None:
        highfreq = samplerate / 2

    features = []

    # Extract MFCC features
    mfcc_feat = mfcc(
                    signal,
                    samplerate,
                    winlen/1000,
                    winstep/1000,
                    numcep,
                    nfilt,
                    nfft,
                    lowfreq,
                    highfreq,
                    appendEnergy=appendEnergy,
                    winfunc=winfunc                
                    )
    mfcc_feat = mfcc_feat.T
    features.append(mfcc_feat)

    # Extract Delta features
    for i in range(delta_order):

        features.append(delta(features[-1], delta_window))

    # Full feature vector
    full_feat = np.vstack(features)

    return full_feat


def load_and_preprocess_data(file_name, file_label, data_path_=data_path, apply_background_noise=False, noise_dict=None, noise_reduction=0.5, features=1, resize=False):
    '''
    features:
    - 1 for MFCC features (default), delta_order=2
    - 2 for log Mel-filterbank energy features
    - 3 for spectrogram
    - 4 for Discrete Wavelet Transform + MFCC features
    - 5 for MFCC features, delta_order=0
    '''
    
    # load data
    data = load_data(file_name, file_label, data_path_=data_path_)
    
    # padding/trimming
    data = padding_trimming(data)
    
    # add background noise
    if apply_background_noise:
        data = background_noise(data, noise_dict=noise_dict, noise_reduction=noise_reduction)

    # extract features
    if features == 1:
        data_features = get_mfcc(data)
    
    elif features == 2:
        data_features = get_logfbank(data)

    elif features == 3:
        data_features = get_spectrogram(data)

    elif features == 4:
        data, _ = dwt(data=data, wavelet='db1', mode='sym')
        data_features = get_mfcc(data)
        
    elif features == 5:
        data_features = get_mfcc(data, delta_order=0)
        
    else:
        data_features = data
    
    # resize feature vector
    if resize:
        data_features = np.resize(data_features, (50, 50))
        
    # TensorFlow takes as input 32-bit floating point data
    return data_features.astype(np.float32)

def remove_file_starting_with(name):
    for filename in glob.glob(name+'*'):
        os.remove(filename) 
        
def reimport_module(module_name):
    importlib.reload(module_name)