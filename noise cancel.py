import os
import pandas as pd
import numpy as np
import scipy
import librosa
from pydub import AudioSegment
from pydub.playback import play
import matplotlib
import numpy
from pydub.utils import which
import matplotlib.pyplot as plt
import numpy as np
import wave, sys
import audioread
# import file
from scipy.io.wavfile import write
from scipy.signal import butter,filtfilt

def cancel(pathn,paths): 
   
    noi = AudioSegment.from_file(pathn, format="wav")
    
    sig=AudioSegment.from_file(paths, format="wav")

    AudioSegment.converter = which("ffmpeg")
    

    def duration_detector(file):
        totalsec=0
        with audioread.audio_open(file) as f:

            # totalsec contains the length in float
            totalsec = f.duration
        length=int(totalsec)
        hours = length // 3600  # calculate in hours
        length %= 3600
        mins = length // 60  # calculate in minutes
        length %= 60
        seconds = length  # calculate in seconds

        return hours, mins, seconds

  
    x,y,z=duration_detector(paths)
    a,b,c=duration_detector(pathn)
    
    
    while x>a or y>b or z>c:
        
        noi+=noi
        if x<a or (x==a and y<b) or (x==a and y==b and z<c):
            break
        if c>=60:
            b+=1
        elif b>=0:
            a+=1
        else:
            a+=a
            b+=b
            c+=c
       
    noi+=noi

    
    invertednoise = noi.invert_phase()
    combined = sig.overlay(invertednoise,position=0)
    name="output.wav"
    combined.export(name, format="wav")
    return name



import os
import scipy.signal as signal
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment

base_path = os.getcwd()
output_path = "output.wav"


def resample(input_signal, old_sample_rate, new_sample_rate):
    resampled_signal = signal.resample_poly(input_signal, new_sample_rate, old_sample_rate)
    return resampled_signal.astype(input_signal.dtype)


def stft(audio, dimension):
    dimensions = audio.ndim
    transform = np.array([], ndmin=1)
    if (dimensions==1):
        transform = librosa.stft(audio) #mono
    else:
        transform = librosa.stft(audio[:,dimension]) #stereo
    return transform


def spectral_subtraction(noise_profile_n, input_signal_y, dimension,level):
    N = stft(noise_profile_n, dimension)
    mN = np.abs(N) # magnitude spectrum

    Y = stft(input_signal_y, dimension)
    mY = np.abs(Y)
    pY = np.angle(Y) # phase spectrum
    poY = np.exp(1.0j * pY) # phase information

    # spectral subtraction
    noise_mean = np.mean(mN, axis=1, dtype="float64") #find noise mean
    noise_mean=  noise_mean.reshape(( noise_mean.shape[0],1))
#     noise_mean = noise_mean[:, np.newaxis] # perform subtraction
    output_X = mY - level*(noise_mean)
    X = output_X
    X = np.clip(output_X, a_min=0.0, a_max=None)

    # add phase info
    X = X * poY

    # inverse STFT
    output_x = librosa.istft(X)
    return output_x


def run(noise_profile, noisy_input, FS,level):
    # read noise profile, input and find dimensions
    n, fs_n = sf.read(noise_profile)
    y, fs_y = sf.read(noisy_input)
    profile_dimensions = n.ndim
    input_dimensions = y.ndim

    if (fs_n != FS):
        n = resample(n, fs_n, FS)
    if (fs_y != FS):
        y = resample(y, fs_y, FS)

    assert profile_dimensions <= 2, "Only mono and stereo files supported."
    assert input_dimensions <= 2, "Only mono and stereo files supported."


    if (profile_dimensions>input_dimensions):
        # make noisy input stereo file
        num_channels = profile_dimensions
        y = np.array([y, y], ndmin=num_channels)
        y = np.moveaxis(y, 0, 1)
    else:
        # make noise profile stereo file
        num_channels = input_dimensions 
        if (profile_dimensions!=input_dimensions):
            n = np.array([n, n], ndmin=num_channels)
            n = np.moveaxis(n, 0, 1)

    # find output for each channel
    for channel in range(num_channels):
        single_channel_output = spectral_subtraction(n, y, channel,level)
        if (channel==0):
            output_x = np.zeros((len(single_channel_output), num_channels))
        output_x[:,channel] = single_channel_output

    # convert to mono
    if (num_channels > 1):
        output_x = np.moveaxis(output_x, 0, 1)
        output_x = librosa.to_mono(output_x)

    # write to wav
    
    sf.write(output_path, output_x, FS, format="WAV")
#     song = AudioSegment.from_wav(output_path) + 2
#     song.export(output_path, format='wav')
    return output_path

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def save_wav (wav,fs, path="out.wav"):
    write(path, fs, wav.astype(np.int16))
    return wav


def lp_filter(data,rate):
    T = 5.0         
    fs = 30.0       
    cutoff = 2  
    
    order = 2      
    n = int(T * fs) 
    
    y=butter_lowpass_filter(data, cutoff, fs, order)
    
    save_wav(y,rate)
    return y
    