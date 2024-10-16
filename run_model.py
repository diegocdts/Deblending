import torch.nn as nn
import torch.optim as optim

from model.linear import Linear
from model.mlp import MLP
from util.arguments import args
from util.config import load
from util.data import ImageTensor
from util.training_validation import CrossValidation

pwd = '/home/src'
args = args()
data_parameters = load(f'{pwd}/config/data.json',args.data)
model_parameters = load(f'{pwd}/config/model.json',args.model)

imageTensor = ImageTensor(data_parameters["root"])
inputs = imageTensor.load_stack(data_parameters["input_file_name"], shape=data_parameters["shape"])
targets = imageTensor.load_stack(data_parameters["output_file_name"], shape=data_parameters["shape"])

lr = model_parameters["lr"]
n_splits = model_parameters["n_splits"]
n_epochs = model_parameters["n_epochs"]
batch_size = model_parameters["batch_size"]
dropout_prob = model_parameters["dropout_prob"]

if args.model == 'linear':
    model = Linear(imageTensor.shape, dropout_prob=dropout_prob)
else:
    hidden_factor = model_parameters["hidden_factor"]
    num_hidden_layers = model_parameters["num_hidden_layers"]

    model = MLP(imageTensor.shape,
                dropout_prob=dropout_prob, hidden_factor=hidden_factor, num_hidden_layers=num_hidden_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

cross_validation = CrossValidation(model, criterion, optimizer, inputs, targets, n_splits, n_epochs, batch_size)
cross_validation.run()
cross_validation.predict(data_parameters["output_path"])
