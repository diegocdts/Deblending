import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, shape: tuple, num_hidden_layers: int, dropout_prob: float = 0.2):
        """
        This class implements a Multi Layer Perceptron (MLP) model with a dropout layer and ReLU activation function
        between the hidden layers.
        :param shape: data shape
        :param num_hidden_layers: number of hidden layers
        :param dropout_prob: dropout probability. Default: 0.2
        """
        super(MLP, self).__init__()
        self.shape = shape

        layers = self.__get_layers__(num_hidden_layers, dropout_prob)

        self.sequential = nn.Sequential(*layers)

    def __get_layers__(self, num_hidden_layers, dropout_prob):
        dim = self.shape[0] * self.shape[1]
        layers = [nn.Flatten(), nn.Dropout(p=dropout_prob), nn.Linear(dim, dim)]

        for _ in range(num_hidden_layers - 1):

            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim, dim))
        return layers

    def forward(self, x):
        x = self.sequential(x)
        return x.view(-1, 1, *self.shape)
