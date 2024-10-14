import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, shape: tuple, num_hidden_layers: int, dropout_prob: float = 0.2, hidden_factor: float =0.5):
        """
        This class implements a Multi Layer Perceptron (MLP) model with a dropout layer and ReLU activation function
        between the hidden layers.
        :param shape: data shape
        :param num_hidden_layers: number of hidden layers
        :param dropout_prob: dropout probability. Default: 0.2
        :param hidden_factor: layer size increase factor. Default: 0.5
        """
        super(MLP, self).__init__()
        self.shape = shape

        layers = self.__get_layers__(num_hidden_layers, dropout_prob, hidden_factor)

        self.sequential = nn.Sequential(*layers)

    def __get_layers__(self, num_hidden_layers, dropout_prob, hidden_factor):
        dim = self.shape[0] * self.shape[1]
        out_hidden_dim = int(dim * hidden_factor)
        layers = [nn.Flatten(), nn.Dropout(p=dropout_prob), nn.Linear(dim, out_hidden_dim)]

        for _ in range(num_hidden_layers - 1):
            in_hidden_dim = out_hidden_dim
            out_hidden_dim = int(in_hidden_dim * hidden_factor)
            layers.append(nn.Linear(in_hidden_dim, out_hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(out_hidden_dim, dim))
        return layers

    def forward(self, x):
        x = self.sequential(x)
        return x.view(-1, 1, *self.shape)
