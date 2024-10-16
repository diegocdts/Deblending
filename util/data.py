import os
import sys

import numpy as np
import segyio
import torch
import pandas as pd
import matplotlib.pyplot as plt


def normalize(tensor: torch.tensor):
    """
    This function applies z-score normalization to a tensor
    :param tensor: the tensor to be normalized
    :return: the normalized tensor
    """
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std


def plot_data(data: np.array, shape: tuple = None):
    """
    This function plots seismic data in image form.
    :param data: data to plot
    :param shape: the shape of the data
    """
    if shape is not None:
        delta_time = 0.002  # sampling interval (time - seconds)
        delta_receptor = 25  # sampling interval (space - m)
        num_receptors = shape[1]
        num_samples = shape[2]
        time_scope = delta_time * np.arange(num_samples)
        space_scope = np.arange(0, num_receptors * delta_receptor, delta_receptor)

        extent = [space_scope[0], space_scope[-1], time_scope[0], time_scope[-1]]
    else:
        extent = None
    plt.imshow(data, aspect='auto', cmap='seismic', origin='upper', vmin=-100, vmax=100, extent=extent)
    plt.colorbar(label='Valor')
    plt.show()


class ImageTensor:

    def __init__(self, root: str):
        self.root = root
        self.shape = None

    def load_tensor_csv(self, file_path: str):
        """
        This method converts seismic data from a csv file to tensor
        :param file_path: path of the seismic data
        :return: tensor data
        """
        try:
            data = pd.read_csv(file_path, header=None).values
        except FileNotFoundError as error:
            print(error)
            sys.exit()
        #plot_data(data.T)
        self.shape = data.shape
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return normalize(tensor)

    def load_tensor(self, tensor_data: np.array):
        """
        This method converts seismic data to tensor
        :param tensor_data: seismic data
        :return: tensor data
        """
        self.shape = tensor_data.shape
        tensor = torch.tensor(tensor_data.real, dtype=torch.float32).unsqueeze(0)
        return normalize(tensor)

    def load_stack_csv(self, data_dir: str):
        """
        This method loads and returns a stack of tensors from csv files
        :param data_dir: data file directory
        :return: image tensor stack
        """
        path = str(os.path.join(self.root, data_dir))
        file_paths = [f'{path}/{file_name}' for file_name in os.listdir(path) if file_name.endswith('csv')]
        return torch.stack([self.load_tensor_csv(file_path) for file_path in file_paths])

    def load_stack(self, file_name: str, shape: tuple, truncated_shape: tuple, final_shape: tuple):
        """
        This method loads and returns a stack of tensors from segy or npy files
        :param file_name: data file name
        :param shape: shape of the data in the form of number of shots, number of receivers and number of samples
        :param truncated_shape: truncated (new) shape of data in the form of number of shots, number of receivers and number of samples
        :param final_shape: the shape of the tensors in the stack
        :return: common receiver image tensor stack
        """
        path = str(os.path.join(self.root, file_name))
        try:
            if 'segy' in file_name:
                f1 = segyio.open(path, ignore_geometry=True)
                data = segyio.collect(f1.trace[:])
            else:
                data = np.load(path)
            data = data.reshape(*shape)
            data = data[:truncated_shape[0], :truncated_shape[1], :truncated_shape[2]]
            data = data.reshape(*final_shape)
        except FileNotFoundError as error:
            print(f'[ERROR] {error}')
            sys.exit()
        except TypeError as error:
            print(f'[ERROR] {error}')
            sys.exit()

        num_receptors = final_shape[1]

        #plot_data(data[:,1,:].real.T, shape)
        return torch.stack([self.load_tensor(data[:, receptor, :]) for receptor in range(num_receptors)])
