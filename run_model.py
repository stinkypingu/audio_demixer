import prepare_dataset
import architecture_fix

import musdb
import librosa
import sounddevice as sd
from pydub import AudioSegment
import soundfile as sf

import numpy as np
import pickle
import time
import torch
from torch.utils.data import DataLoader

FRAME_SIZE = 2048
HOP_LENGTH = 512

DATA_PATH = r'C:\Projects\TBP\music_generator\data\musdb'
BATCH_SIZE = 32

IN_CHANNELS = 1
OUT_CHANNELS = 1

MODEL_FILENAME = 'model_20epoch_4subset_16batchsize'
#MODEL_FILENAME = 'model_50epoch_25songs'

AUDIO_FILENAME = 'hyolyn_i_choose_to_love_you'
AUDIO_FILENAME = 'bruno_die_with_a_smile'
AUDIO_FILENAME = 'adele_someone_like_you'
AUDIO_FILENAME = 'sinatra_my_way'

#AUDIO_FILENAME = 'sometimes'


model = architecture_fix.UNet(IN_CHANNELS, OUT_CHANNELS)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

checkpoint = torch.load(f'{MODEL_FILENAME}.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f'successfully loaded model from checkpoint: {MODEL_FILENAME}, loss: {checkpoint["final_loss"]}')

model.eval()


start_time = time.time()
audio_file = f'data/{AUDIO_FILENAME}.mp3'
y, sr = librosa.load(audio_file)
y = np.repeat(y, 2)
sr = sr * 2

'''sd.play(y, sr)
input('press enter to stop playback')
sd.stop()'''




'''stfts, input_segments = prepare_dataset.segment(y, sr, segment_length=2)

song_data = prepare_dataset.SpectrogramDataset(input_segments)

s1 = [song_data[idx] for idx in range(len(song_data))]
print(s1[0].shape)

s2 = prepare_dataset.reconstruct_spectrograms(s1)
print(s2[0].shape)

audio1, sr1 = prepare_dataset.reconstruct_audio(stfts, input_segments)
sd.play(audio1, sr1)
input('press enter to stop playback')
sd.stop()'''


end_time = time.time()

print(f'successfully processed song, time: {end_time - start_time :.2f}')

stfts, input_segments = prepare_dataset.segment(y, sr, segment_length=2)

song_data = prepare_dataset.SpectrogramDataset(input_segments)

dataloader = DataLoader(song_data, batch_size=16, shuffle=False)
end_time = time.time()

print(f'successfully loaded song: {AUDIO_FILENAME}, time: {end_time - start_time :.2f}')



start_time = time.time()
outputs = []
with torch.no_grad():
    for inputs in dataloader:
        preds = model(inputs)

        print(f'output shape: {preds.shape}')

        outputs.append(preds)

spectrograms = prepare_dataset.reconstruct_spectrograms(outputs)

audio, sr = prepare_dataset.reconstruct_audio(stfts, spectrograms)
end_time = time.time()

print(f'successfully processed song, time: {end_time - start_time :.2f}')



print(audio.shape)
print(sr)

def normalize_sine_wave(wave):
    norm = wave / np.max(np.abs(wave))
    return norm

#blank = np.zeros_like(y)
#blank[:len(audio)] = audio

def smooth_audio(audio, kernel_size=10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(audio, kernel, mode='same')

from scipy.signal import butter, filtfilt

def low_pass_filter(audio, sr, cutoff=4000):
    nyquist = sr / 2
    norm_cutoff = cutoff / nyquist
    b, a = butter(4, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)

#smoothed_audio = low_pass_filter(audio, sr)

#smoothed_audio = smooth_audio(audio)
audio = normalize_sine_wave(audio)



x = len(audio)
sf.write(f'{AUDIO_FILENAME}_AWFUL_INSTR.wav', audio, sr)
#sf.write(f'{AUDIO_FILENAME}_AWFUL.wav', audio[int(0.5*x):int(0.7*x)], sr)

sd.play(audio, sr)
input('press enter to stop playback')
sd.stop()
