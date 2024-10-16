import os
import sys

import numpy as np
import segyio
import torch
import matplotlib.pyplot as plt


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


def get_mean_std(tensor: torch.tensor):
    """
    returns the mean and std of a tensor
    :param tensor: the tensor to get the mean and std from
    :return: mean and std
    """
    return tensor.mean(), tensor.std()

def normalize(tensor: torch.tensor, mean: float, std: float):
    """
    applies the z-score normalization on a tensor
    :param tensor: the tensor to be normalized
    :param mean: mean of the tensor
    :param std: std of the tensor
    :return: the normalized tensor
    """
    return (tensor - mean) / std

def un_normalize(tensor: torch.tensor, mean: float, std: float):
    """
    undo the z-score normalization on a tensor
    :param tensor: the tensor to be un-normalized
    :param mean: mean of the tensor
    :param std: std of the tensor
    :return: the un-normalized tensor
    """
    return (tensor * std) + mean

class ImageTensor:

    def __init__(self, root: str,
                 input_path: str, target_path: str,
                 shape: tuple, truncated_shape: tuple, final_shape: tuple):
        """
        This method loads and returns a stack of tensors from segy or npy files
        :param root: root directory
        :param input_path: input file name
        :param target_path: target file name
        :param shape: shape of the data in the form of number of shots, number of receivers and number of samples
        :param truncated_shape: truncated (new) shape of data in the form of number of shots, number of receivers and number of samples
        :param final_shape: the shape of the tensors in the stack
        :return: common receiver image tensor stack
        """
        self.root = root
        self.input_path = input_path
        self.target_path = target_path
        self.shape = shape
        self.truncated_shape = truncated_shape
        self.final_shape = final_shape
        self.image_shape = None

        self.input = self.load_stack(input_path)
        self.target = self.load_stack(target_path)
        
        self.input_mean, self.input_std = get_mean_std(self.input)
        self.target_mean, self.target_std = get_mean_std(self.target)

    def load_tensor(self, tensor_data: np.array):
        """
        This method converts seismic data to tensor
        :param tensor_data: seismic data
        :return: tensor data
        """
        self.image_shape = tensor_data.shape
        tensor = torch.tensor(tensor_data.real, dtype=torch.float32).unsqueeze(0)
        return tensor

    def load_stack(self, file_name: str):
        path = str(os.path.join(self.root, file_name))
        try:
            if 'segy' in file_name:
                f1 = segyio.open(path, ignore_geometry=True)
                data = segyio.collect(f1.trace[:])
            else:
                data = np.load(path)
            data = data.reshape(*self.shape)
            data = data[:self.truncated_shape[0], :self.truncated_shape[1], :self.truncated_shape[2]]
            data = data.reshape(*self.final_shape)
        except FileNotFoundError as error:
            print(f'[ERROR] {error}')
            sys.exit()
        except TypeError as error:
            print(f'[ERROR] {error}')
            sys.exit()

        num_receptors = self.final_shape[1]

        #plot_data(data[:,1,:].real.T, shape)
        return torch.stack([self.load_tensor(data[:, receptor, :]) for receptor in range(num_receptors)])
