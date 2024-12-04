import prepare_dataset
import architecture_fix

import musdb
import librosa
import numpy as np
import pickle
import gzip
import time
import torch
from torch.utils.data import DataLoader
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FRAME_SIZE = 2048
HOP_LENGTH = 512

DATA_PATH = r'C:\Projects\TBP\music_generator\data\musdb'
BATCH_SIZE = 32

IN_CHANNELS = 1
OUT_CHANNELS = 1

PRE_PROCESSED = False
TRAIN_DATASET_FILENAME = 'all_songs'

USE_CHECKPOINT = True
MODEL_FILENAME = 'model_50songs_4encoder'


os.environ['MUSDB_PATH'] = DATA_PATH
musdb_subsets = musdb.DB(subsets=['train'])

for count, track in enumerate(musdb_subsets):
    print(f'processing song {count+1}/{len(musdb_subsets)}: {track.targets.keys()}')


#compact zip
def build(path, subsets, target='vocals'):

    os.environ['MUSDB_PATH'] = path
    musdb_subsets = musdb.DB(subsets=subsets)

    for count, track in enumerate(musdb_subsets):
        print(f'processing song {count+1}/{len(musdb_subsets)}: {track.name}')


        t1 = time.time()
        #original mix
        y_input = np.mean(track.audio, axis=1)

        #vocals
        y_target = np.mean(track.sources[target].audio, axis=1)

        assert(len(y_input) == len(y_target))
        
        combined = np.stack([y_input, y_target], axis=0)
        t2 = time.time()

        gzipped_file = f'data/compact_musdb/track_{count+1}.npy.gz'
        with gzip.GzipFile(gzipped_file, 'wb') as file:
            np.save(file, combined)
        t3 = time.time()
        print(f'finished {count+1}/{len(musdb_subsets)}, extraction time: {t2 - t1 :.2f}, compression time: {t3 - t2 :.2f}, shape: {combined.shape}\n')

    return

#build(DATA_PATH, ['train', 'test'], target='vocals')



'''
#process a dataset to use and save it, or use one that has already been saved
inputs, targets = None, None
if not PRE_PROCESSED:
    print(f'pre-processing...')

    start_time = time.time()

    for i in range(15):
        print(f'\nprocessing songs {(i * 10) + 1}-{(i * 10) + 10}')
        inputs, targets = prepare_dataset.build(DATA_PATH, subsets=['train', 'test'], target='vocals', segment_length=2, start_idx=(i*10), num_songs=10)

        #with gzip.open(f'data/processed_musdb/{TRAIN_DATASET_FILENAME}_{i}.pkl.gz', 'wb') as file:
        #    pickle.dump((inputs, targets), file)
    end_time = time.time()

    print(f'successfully saved to {TRAIN_DATASET_FILENAME}.pkl, time: {end_time - start_time :.2f}')
'''


