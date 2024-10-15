import numpy as np
import torch

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class CrossValidation:

    def __init__(self, model,
                 criterion,
                 optimizer,
                 inputs,
                 targets,
                 n_splits: int,
                 n_epochs: int,
                 batch_size: int):
        """
        Implements cross-validation
        :param model: model to be trained and validated
        :param criterion: loss function
        :param optimizer: optimizer
        :param inputs: input tensors
        :param targets: target tensors
        :param n_splits: number of cross-validation splits (kfold)
        :param n_epochs: number of epochs
        :param batch_size: batch size
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.inputs = inputs
        self.targets = targets
        self.k_fold = KFold(n_splits=n_splits)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.losses = []
        self.best_model = None
        self.lowest_loss = float('inf')

    def run(self):
        """
        Performs cross-validation and displays the average validation loss
        """
        self.losses = []
        for split_index, (train_index, val_index) in enumerate(self.k_fold.split(self.inputs)):
            print(f'Cross Validation split {split_index + 1}')

            x_train, x_val = self.inputs[train_index], self.inputs[val_index]
            y_train, y_val = self.targets[train_index], self.targets[val_index]

            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False)

            model = self.model
            optimizer = self.optimizer

            for epoch in range(self.n_epochs):
                model, optimizer, train_loss = self.__train_epoch__(model, train_loader, optimizer)
                val_loss = self.__validate__(model, val_loader)
                if val_loss < self.lowest_loss:
                    self.lowest_loss = val_loss
                    self.best_model = model
                print(f'Epoch {epoch + 1}/{self.n_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
                if epoch == self.n_epochs - 1:
                    self.losses.append(val_loss)
        mean_loss = sum(self.losses) / len(self.losses)
        print('Cross validation concluded with mean loss =', mean_loss)


    def __train_epoch__(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return model, optimizer, total_loss / len(train_loader)


    def __validate__(self, model, val_loader):
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                val_outputs = model(x_val_batch)
                val_loss = self.criterion(val_outputs, y_val_batch)
                val_loss_total += val_loss.item()
        return val_loss_total / len(val_loader)

    def predict(self, output_path):
        model = self.best_model.eval()
        inputs = self.inputs
        with torch.no_grad:
            outputs = model(inputs)
            outputs_array = outputs.cpu().numpy()
            np.save(output_path, outputs_array)
