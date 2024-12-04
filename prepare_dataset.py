import numpy as np
import musdb
import librosa
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FRAME_SIZE = 2048
HOP_LENGTH = 512
DATA_PATH = r'C:\Projects\TBP\music_generator\data\musdb'


#dataset
class SpectrogramDataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets #None during inference

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        #gets shape (1025, 173)
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)

        #adds channel dimension (1, 1025, 173)
        input_data_channel = input_data.unsqueeze(0)

        #divisible by 2^4 for 4 maxpools (1, 1040, 176)
        input_data_padded = F.pad(input_data_channel, (0, 3, 0, 15), mode='constant', value=0)

        #training mode includes targets
        if self.targets is not None:
            target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
            target_data_channel = target_data.unsqueeze(0)
            target_data_padded = F.pad(target_data_channel, (0, 3, 0, 15), mode='constant', value=0)

            return input_data_padded, target_data_padded
        
        #inference mode
        return input_data_padded


#cut a song into spectrograms of segments
def segment(y, sr, segment_length=2):

    #number of segments, ignore the last block because it won't be long enough
    chunk_size = segment_length * sr
    num_segments = int(np.ceil(len(y) / chunk_size)) - 1

    stfts = []
    spectrograms = []
    for i in range(0, num_segments * chunk_size, chunk_size):
        segment = y[i : i + chunk_size]

        #short-time fourier transform
        stft = librosa.stft(segment, n_fft=FRAME_SIZE, win_length=FRAME_SIZE, hop_length=HOP_LENGTH)
        stfts.append(stft)

        #power spectrogram
        power = np.abs(stft) ** 2
        
        spectrogram = librosa.power_to_db(power)

        spectrograms.append(spectrogram)
    return stfts, spectrograms



#builds a dataset into the right format
def build(path, subsets, target='vocals', segment_length=2, start_idx=0, num_songs=10):

    os.environ['MUSDB_PATH'] = path
    musdb_subsets = musdb.DB(subsets=subsets)

    inputs = []
    targets = []

    for count, track in enumerate(musdb_subsets[start_idx : start_idx + num_songs]):
        print(f'processing song {count+1}/{num_songs}: {track.name}')

        #original mix
        y_input = np.mean(track.audio, axis=1)
        sr_input = track.rate

        #vocals
        y_target = np.mean(track.sources[target].audio, axis=1)
        sr_target = track.sources[target].rate

        assert(len(y_input) == len(y_target))
        assert(sr_input == sr_target)
            
        _, input_segments = segment(y_input, sr_input, segment_length=segment_length)
        _, target_segments = segment(y_target, sr_target, segment_length=segment_length)

        assert(len(input_segments) == len(target_segments))

        inputs.extend(input_segments)
        targets.extend(target_segments)

        print(input_segments)
    return inputs, targets



#given output list of tensors returns it to a list of unpadded spectrograms
def reconstruct_spectrograms(tensor_list):

    spectrograms = []
    for tensor in tensor_list:
        batch_size = tensor.size(0)

        for i in range(batch_size):
            #shape (1, 1040, 176)
            padded_3d = tensor[i]

            #shape (1040, 176)
            padded_2d = padded_3d.squeeze(0)

            #shape (1025, 173)
            unpadded = padded_2d[:-15, :-3]

            #convert to numpy
            unpadded_np = unpadded.detach().cpu().numpy()

            spectrograms.append(unpadded_np)

    return spectrograms


#effectively reverse the cutting up of a song waveform into segments
def reconstruct_audio(stfts, spectrograms, sr=44100, segment_length=2):
    reconstructed = []
    
    stfts = np.hstack(stfts)
    spectrograms = np.hstack(spectrograms)
    mag = librosa.db_to_amplitude(spectrograms)

    phase = librosa.magphase(stfts)[1]
    complex = mag * phase

    reconstruct = librosa.istft(complex, win_length=FRAME_SIZE, hop_length=HOP_LENGTH)
    return reconstruct, sr
    '''
    for i in range(len(spectrograms)):

        phase = librosa.magphase(stfts[i])[1]
        spectrogram = spectrograms[i]

        mag = librosa.db_to_amplitude(spectrogram)

        complex_spectrogram = mag * phase

        segment_audio = librosa.istft(complex_spectrogram, win_length=FRAME_SIZE, hop_length=HOP_LENGTH)

        reconstructed.append(segment_audio)
    '''
    
    reconstructed = np.concatenate(reconstructed)
    return reconstructed, sr







'''track = mus_train[0]
y = np.mean(track.audio, axis=1)
sr = track.rate

print(len(y))
print(sr)

x = segment(y, sr)

print(len(x))
print(x[0].shape)

for track in mus_train:
    #original mix
    y = np.mean(track.audio, axis=1)
    sr = track.rate

    input = segment(y, sr)

    #target vocal track
    y2 = np.mean(track.sources['vocals'].audio, axis=1)
    sr2 = track.sources['vocals'].rate

    target = segment(y2, sr2)

    assert(len(y) == len(y2))
    assert(len(input) == len(target))
'''
