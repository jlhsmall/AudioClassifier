
import librosa
audio_path = 'data/UrbanSound8K/audio/fold7/101848-9-0-0.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))

#%%

import IPython.display as ipd
ipd.Audio(audio_path)

#%%

#display waveform
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(10, 4))
librosa.display.waveplot(x, sr=sr)


S = librosa.feature.melspectrogram(y=x, sr=sr)

import numpy as np

plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
print(S_dB.shape)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
#plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()



S = np.abs(librosa.stft(x))
contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
print(contrast.shape)

plt.figure(figsize=(10,4))
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar()
plt.ylabel('Frequency bands')
#plt.title('Spectral contrast')
plt.tight_layout()
plt.show()



chroma = librosa.feature.chroma_stft(S=S, sr=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
#plt.title('Chromagram')
plt.tight_layout()
plt.show()



mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
#plt.title('MFCC')
plt.tight_layout()
plt.show()
