import os.path

import torch
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from util.data import normalize, un_normalize

torch.cuda.empty_cache()

class CrossValidation:

    def __init__(self,
                 model,
                 model_name,
                 criterion,
                 optimizer,
                 imageTensor,
                 n_splits: int,
                 n_epochs: int,
                 batch_size: int,
                 outputs_path: str):
        """
        Implements cross-validation
        :param model: model to be trained and validated
        :param model_name: name of the model
        :param criterion: loss function
        :param optimizer: optimizer
        :param imageTensor: ImageTensor object
        :param n_splits: number of cross-validation splits (k-fold)
        :param n_epochs: number of epochs
        :param batch_size: batch size
        :param outputs_path: output directory
        """
        self.model = torch.nn.DataParallel(model)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.imageTensor = imageTensor
        self.input = normalize(imageTensor.input, imageTensor.input_mean, imageTensor.input_std)
        self.target = normalize(imageTensor.target, imageTensor.target_mean, imageTensor.target_std)
        self.k_fold = KFold(n_splits=n_splits)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.best_model = None
        self.lowest_loss = float('inf')
        self.outputs_path = outputs_path.replace('_XX_', f'_{model_name}_')

        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)

    def run(self):
        """
        Performs cross-validation and displays the average validation loss
        """

        for split_index, (train_index, val_index) in enumerate(self.k_fold.split(self.input)):
            print(f'Cross Validation split {split_index + 1}')

            train_losses = []
            val_losses = []

            x_train, x_val = self.input[train_index], self.input[val_index]
            y_train, y_val = self.target[train_index], self.target[val_index]

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
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            self.write_losses(train_losses, val_losses, split_index+1)
        print('Cross validation concluded!')


    def __train_epoch__(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
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
                x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)
                val_outputs = model(x_val_batch)
                val_loss = self.criterion(val_outputs, y_val_batch)
                val_loss_total += val_loss.item()
        return val_loss_total / len(val_loader)

    def predict(self):
        model = self.best_model.eval()
        inputs = self.input.to(self.device)
        with torch.no_grad():
            outputs = un_normalize(model(inputs), self.imageTensor.target_mean, self.imageTensor.target_std)
            outputs_array = outputs.cpu().numpy()
            np.save(f'{self.outputs_path}/predicted_deblended_{self.model_name}.npy', outputs_array)

    def write_losses(self, train, val, split_index):
        losses = np.column_stack((train, val))
        np.savetxt(f'{self.outputs_path}/losses_{self.model_name}_split{split_index}.csv', losses, header='Train loss Val loss', fmt='%.5f')