import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, units: int, num_hidden_layers: int, dropout_prob: float = 0.2):
        """
        This class implements a Multi Layer Perceptron (MLP) model with a dropout layer and ReLU activation function
        between the hidden layers.
        :param units: input layer units
        :param num_hidden_layers: number of hidden layers
        :param dropout_prob: dropout probability. Default: 0.2
        """
        super(MLP, self).__init__()
        self.units = units

        layers = self.__get_layers__(num_hidden_layers, dropout_prob)

        self.sequential = nn.Sequential(*layers)

    def __get_layers__(self, num_hidden_layers, dropout_prob):
        layers = [nn.Flatten(), nn.Dropout(p=dropout_prob)]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(self.units, self.units))
            if num_hidden_layers > 2:
                layers.append(nn.ReLU())

        layers.append(nn.Linear(self.units, self.units))
        return layers

    def forward(self, x):
        x = self.sequential(x)
        return x
