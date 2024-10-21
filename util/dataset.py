import glob

import numpy as np
import segyio
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def mean_std(data: np.array):
    return data.mean(), data.std()

def normalize_z_score(data, mean, std):
    return (data - mean) / std

def denormalize_z_score(normalized_data, mean, std):
    return (normalized_data * std) + mean

def plot_image(data):
    plt.imshow(data.T, aspect='auto', cmap='seismic', origin='upper', vmin=-100, vmax=100)
    plt.colorbar()
    plt.show()

def load_file(file_path):
    if 'sgy' in file_path:
        f1 = segyio.open(file_path, ignore_geometry=True)
        data = segyio.collect(f1.trace[:])
    else:
        data = np.load(file_path)
    return data.real

def load_data(path, train_percent, val_percent):
    files = sorted(glob.glob(f'{path}/*'))
    train_end = len(files) * train_percent
    val_end = (len(files) * val_percent) + train_end
    data_train, data_val, data_test = [], [], []

    for index, file_path in enumerate(files):
        samples = load_file(file_path)
        if index < int(train_end):
            data_train.extend(samples)
        elif index < int(val_end):
            data_val.extend(samples)
        else:
            data_test.extend(samples)
    return np.array(data_train), np.array(data_val), np.array(data_test)

class ImageDataset(Dataset):

    def __init__(self, input_data: np.array, target_data: np.array):
        self.input_images = input_data
        self.target_images = target_data

        self.input_images = torch.FloatTensor(self.input_images).unsqueeze(1)
        self.target_images = torch.FloatTensor(self.target_images).unsqueeze(1)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        target_img = self.target_images[idx]
        return input_img, target_img
