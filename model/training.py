import numpy as np
import torch
import copy
import torch.nn as nn

from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader, Subset

from model.mlp import MLP
from util.dataset import ImageDataset, load_data, mean_std, normalize_z_score, denormalize_z_score


class TrainingValidation:

    def __init__(self, name, model_parameters, data_parameters):
        input_path = data_parameters["input_path"]
        target_path = data_parameters["target_path"]
        shape = data_parameters["shape"]
        outputs_path = data_parameters["outputs_path"]
        n_samples = shape[0]

        num_hidden_layers = model_parameters["num_hidden_layers"]
        dropout_prob = model_parameters["dropout_prob"]
        n_splits = model_parameters["n_splits"]

        input_dataset = load_data(input_path, shape)
        target_dataset = load_data(target_path, shape)
        self.name = name
        self.mean, self.std = mean_std(input_dataset[:int(0.7*n_samples)])

        normalized_input = normalize_z_score(input_dataset, self.mean, self.std)
        normalized_target = normalize_z_score(target_dataset, self.mean, self.std)
        self.train_dataset = ImageDataset(normalized_input, normalized_target, end=int(0.7*n_samples))
        self.val_dataset = ImageDataset(normalized_input, normalized_target, start=int(0.7*n_samples), end=int(0.9*n_samples))
        self.test_dataset = ImageDataset(normalized_input, normalized_target, start=int(0.9 * n_samples))

        self.model = MLP(shape, num_hidden_layers, dropout_prob)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.k_fold = KFold(n_splits=n_splits)
        self.n_epochs = model_parameters["n_epochs"]
        self.batch_size = model_parameters["batch_size"]
        self.learning_rate = model_parameters["lr"]

        self.lowest_loss = float('inf')
        self.outputs_path = outputs_path.replace('_XX_', f'_{self.name}_')

    def train(self):
        """
        Performs cross-validation over the model
        """
        val_losses = []
        for fold, (train_index, val_index) in enumerate(self.k_fold.split(self.train_dataset)):
            train_subset = Subset(self.train_dataset, train_index)
            val_subset = Subset(self.val_dataset, val_index)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            model = copy.deepcopy(self.model)
            model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # training
            model.train()
            for epoch in range(self.n_epochs):
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            #validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_losses.append(val_loss)
            if val_loss < self.lowest_loss:
                self.lowest_loss = val_loss
                torch.save(model.state_dict(), f'{self.outputs_path}/model_weights.pth')
        val_losses = np.array(val_losses)
        self.write_losses(val_losses)

    def predict(self):
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        model = copy.deepcopy(self.model)
        model.to(self.device)
        model.load_state_dict(torch.load(f'{self.outputs_path}/model_weights.pth'))
        model.eval()

        predictions = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                outputs = model(x_batch)
                outputs = denormalize_z_score(outputs.cpu().numpy(), self.mean, self.std)
                predictions.append(outputs)
        predictions = np.array(predictions)
        np.save(f'{self.outputs_path}/prediction_{self.name}.npy', predictions)

    def write_losses(self, losses):
        np.savetxt(f'{self.outputs_path}/losses_{self.name}.csv', losses, fmt='%.5f')