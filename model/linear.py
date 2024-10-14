import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, shape: tuple, dropout_prob: float = 0.2):
        """
        This class implements a simple linear model with one dropout layer.
        :param shape: data shape
        :param dropout_prob: dropout probability. Default: 0.2
        """
        super(Linear, self).__init__()
        dim = shape[0] * shape[1]
        self.shape = shape
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x.view(-1, 1, *self.shape)
