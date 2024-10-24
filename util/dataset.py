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

def load_data(input_path, target_path, start, end):
    input_files = sorted(glob.glob(f'{input_path}/*'))[start:end]
    target_files = sorted(glob.glob(f'{target_path}/*'))
    for input_file in input_files:
        suffix = input_file[-20:]
        matching_targets = [target_file for target_file in target_files if target_file.endswith(suffix)]
        target_files = matching_targets

    data_input, data_target = [], []

    for index, _ in enumerate(input_files):
        input_samples = load_file(input_files[index])
        data_input.extend(input_samples)
        target_samples = load_file(target_files[index])
        data_target.extend(target_samples)
    return np.array(data_input), np.array(data_target)

class ImageDataset(Dataset):

    def __init__(self, input_data: np.array, target_data: np.array):
        self.input_images = input_data
        self.target_images = target_data

        self.input_images = torch.FloatTensor(self.input_images)
        self.target_images = torch.FloatTensor(self.target_images)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        target_img = self.target_images[idx]
        return input_img, target_img
