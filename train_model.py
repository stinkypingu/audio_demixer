import prepare_dataset
import architecture_fix

import musdb
import librosa
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

PRE_PROCESSED = False
TRAIN_DATASET_FILENAME = 'training_100songs'

USE_CHECKPOINT = True
MODEL_FILENAME = 'model_50songs_4encoder'

#process a dataset to use and save it, or use one that has already been saved
inputs, targets = None, None
if not PRE_PROCESSED:
    print(f'pre-processing...')

    start_time = time.time()
    inputs, targets = prepare_dataset.build(DATA_PATH, ['train'], max_songs=50)
    with open(f'{TRAIN_DATASET_FILENAME}.pkl', 'wb') as file:
        pickle.dump((inputs, targets), file)
    end_time = time.time()

    print(f'successfully saved to {TRAIN_DATASET_FILENAME}.pkl, time: {end_time - start_time :.2f}')

else:
    print(f'extracting processed data...')
    
    start_time = time.time()
    with open(f'{TRAIN_DATASET_FILENAME}.pkl', 'rb') as file:
        inputs, targets = pickle.load(file)
    end_time = time.time()

    print(f'successfully extracted from {TRAIN_DATASET_FILENAME}.pkl, time: {end_time - start_time :.2f}')

#load in the training 
print(f'\nloading in training data...')

start_time = time.time()
dataset = prepare_dataset.SpectrogramDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
end_time = time.time()

print(f'successfully loaded into dataloader, batches: {len(dataloader)}, time: {end_time - start_time :.2f}')



#start up the model
print(f'\nstarting model...')
model = architecture_fix.UNet(IN_CHANNELS, OUT_CHANNELS)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)

#use a previous save state model
if USE_CHECKPOINT:
    checkpoint = torch.load(f'{MODEL_FILENAME}.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    total_epochs = checkpoint['total_epochs']

    print(f'successfully loaded from checkpoint, final_loss: {checkpoint["final_loss"]}')

else:
    total_epochs = 0
    
    print(f'new model...')


#keep track of epochs and most recent loss of the model
loss = None
run_epochs = 1
batch_print_freq = 5

print(f'\nstarting training for {run_epochs} epochs.')

for epoch in range(run_epochs):
    model.train()
    running_loss = 0.0

    start_time = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        
        #print statement every
        if ((i+1) % batch_print_freq == 0) or (i == len(dataloader)):
            print(f'batch {i+1}/{len(dataloader)}')

        #backprop
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    end_time = time.time()

    loss = running_loss / len(dataloader)

    #debug
    print(f'epoch [{epoch+1}/{run_epochs}], loss: {loss :.2f}, time: {end_time - start_time :.2f}\n')

#save the model
torch.save({
    'status': 'training...',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_loss': loss,
    'total_epochs': total_epochs + run_epochs,
}, f'{MODEL_FILENAME}.pth')