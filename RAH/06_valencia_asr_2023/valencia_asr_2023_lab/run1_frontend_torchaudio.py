import torchaudio
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
"""
# Sample wav files
"""
wav = 'data1/train/train_00001_m_01_0.wav' # https://github.com/soerenab/AudioMNIST
wav = 'wav/audio1.wav'                   # https://en.wikipedia.org/wiki/Singing
wav = 'wav/audio2.wav'                   # https://www.chineasy.com/mandarin-chinese-pronunciation-an-introduction/

"""
# Waveform

Read and plot the waveform of a wav file.
"""
x, fs = torchaudio.load(wav)
print(x.shape)
plt.plot(x.t().numpy())
plt.savefig('out/waveform.png')

"""
# Power spectrogram

Extract the power spectrogram of a wav file with a 25ms window and 10ms shift using the Short-Time Fourier Transform (STFT).
"""
specgram = torchaudio.transforms.Spectrogram(
    n_fft=512, win_length=25*16, hop_length=10*16)(x)       # 25ms window, 10ms shift
print(specgram.shape)
plt.imshow(specgram.log()[0].numpy(), cmap='jet', aspect='auto',origin='lower') 
plt.savefig('out/spectrogram.png')


"""
# Mel-spectrogram

Extract the mel-spectrogram of a wav file with a 25ms window and 10ms shift using the Short-Time Fourier Transform (STFT) and
a mel filterbank with 80 filters.
"""
melspecgram = torchaudio.transforms.MelSpectrogram(
    n_fft=512, win_length=25*16, hop_length=10*16, n_mels=80)(x)       # 25ms window, 10ms shift
print(melspecgram.shape)
plt.imshow(melspecgram.log()[0].numpy(), cmap='jet', aspect='auto',origin='lower')
plt.savefig('out/melspectrogram.png')

"""
# MFCC

Extract the mel-frequency cepstral coefficients (MFCC) of a wav file with a 25ms window and 10ms shift using the Short-Time Fourier Transform (STFT) and
"""
mfcc = torchaudio.transforms.MFCC(
    melkwargs={"n_fft": 25*16, "hop_length": 10*16, "n_mels": 48, "center": False},       # 25ms window, 10ms shift
    n_mfcc=40, log_mels=True)(x)
print(mfcc.shape)
plt.imshow(mfcc[0,1:].numpy(), cmap='jet', aspect='auto',origin='lower')
plt.savefig('out/mfcc.png')

"""
# Real cepstrum

Extract the real cepstrum of a wav file with a 25ms window and 10ms shift using the Short-Time Fourier Transform (STFT).
"""
f, t, cspecgram = scipy.signal.stft(
    x[0].numpy(), fs=fs,
    nperseg=25*16, noverlap=25*16-10*16, nfft=512, return_onesided=False)   # 25ms window, 10ms shift
print(cspecgram.shape, cspecgram.dtype)
ceps = np.fft.ifft(np.log(np.abs(cspecgram)+1e-6), axis=0)
print(ceps.shape)
plt.imshow(np.abs(ceps)[1:256], cmap='jet', aspect='auto',origin='lower') 
plt.savefig('out/cepstrum.png')

