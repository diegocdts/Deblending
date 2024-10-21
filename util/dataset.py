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

def load_file(file_path, shape):
    if 'segy' in file_path:
        f1 = segyio.open(file_path, ignore_geometry=True)
        data = segyio.collect(f1.trace[:])
    else:
        data = np.load(file_path)
    if data.shape != shape:
        data = data.reshape(*shape)
    return data.real

def plot_image(data):
    plt.imshow(data.T, aspect='auto', cmap='seismic', origin='upper', vmin=-100, vmax=100)
    plt.colorbar()
    plt.show()

def load_data(path, shape):
    return load_file(path, shape)

class ImageDataset(Dataset):

    def __init__(self, input_data: np.array, target_data: np.array, start: int = None, end: int = None):
        if start is None:
            start = 0
        if end is None:
            end = len(input_data)
        self.input_images = input_data[start:end]
        self.target_images = target_data[start:end]

        self.input_images = torch.FloatTensor(self.input_images).unsqueeze(1)
        self.target_images = torch.FloatTensor(self.target_images).unsqueeze(1)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        target_img = self.target_images[idx]
        return input_img, target_img
