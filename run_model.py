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
data_name = args.data
model_name = args.model
data_parameters = load(f'{pwd}/config/data.json', data_name)
model_parameters = load(f'{pwd}/config/model.json', model_name)

root = data_parameters["root"]
input_path = data_parameters["input_file_name"]
target_path = data_parameters["target_file_name"]
shape = data_parameters["shape"]
truncated_shape = data_parameters["truncated_shape"]
final_shape = data_parameters["final_shape"]

imageTensor = ImageTensor(root, input_path, target_path, shape, truncated_shape, final_shape)

lr = model_parameters["lr"]
n_splits = model_parameters["n_splits"]
n_epochs = model_parameters["n_epochs"]
batch_size = model_parameters["batch_size"]
dropout_prob = model_parameters["dropout_prob"]
outputs_path = data_parameters["outputs"]

if args.model == 'linear':
    model = Linear(imageTensor.image_shape, dropout_prob=dropout_prob)
else:
    hidden_factor = model_parameters["hidden_factor"]
    num_hidden_layers = model_parameters["num_hidden_layers"]

    model = MLP(imageTensor.image_shape,
                dropout_prob=dropout_prob, hidden_factor=hidden_factor, num_hidden_layers=num_hidden_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

cross_validation = CrossValidation(model, model_name, criterion, optimizer, imageTensor, n_splits, n_epochs, batch_size, outputs_path)
cross_validation.run()
cross_validation.predict()
