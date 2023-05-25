import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

def convert_loader(input_data: np.ndarray, output_data: np.ndarray, batch=32):
    """
    Convert input data (np.ndarray) to torch DataLoader.

    input_data: Train or predict input data feeding to torch model
    output_data: Train or predict output (label) data
    batch: batch size
    """
    input_output = InputOutputSet(input_data, output_data)
    dt_loader = DataLoader(input_output, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

class RegressionCnn1d(nn.Module):
    def __init__(self, input_dim: int, input_len: int, output_dim: int, output_len: int):
        """
        1D-CNN based Regression model.

        input_dim: dimension (channel size) of input data
        input_len: length (window size) of input data
        output_dim: dimension (channel size) of output data
        output_len: length (window size) of output data
        """
        super(RegressionCnn1d, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_dim = output_dim
        self.output_len = output_len
        self._define_variables()
        self._define_layers()

    def _define_variables(self):
        """
        Define model-related variables.
        """
        self._last_conv_channels = 8

    def _define_layers(self):
        """
        Define layers.
        """
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=8, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding="same")
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=self._last_conv_channels, kernel_size=3, padding="same")
        self.fc1 = nn.Linear(self._last_conv_channels*self.input_len, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, self.output_dim*self.output_len)

    def forward(self, x):
        """
        Regress using input data.

        x: input data
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, self._last_conv_channels*self.input_len)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, self.output_dim, self.output_len)
        return x

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        """
        Input-output dataset composer for converting to dataloader.

        input_dt: input data
        output_dt: output data
        """
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class NnTrainer:
    def __init__(self, model_name: str, input_dim: int, input_len: int, output_dim: int, output_len: int):
        """
        Neural network Regression model trainer.

        model_name: model name
        input_dim: dimension (channel size) of input data
        input_len: length (window size) of input data
        output_dim: dimension (channel size) of output data
        output_len: length (window size) of output data
        """
        if model_name == "RegressionCnn":
            self.model = RegressionCnn1d(input_dim, input_len, output_dim, output_len)
            
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        """
        Fit (train) regressor. Use NAdam optimizer.

        train_loader: train data loader
        epoch: train epoch
        """
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_loss = 0
            for train_x, train_y in train_loader:
                train_x = train_x.float()
                train_y = train_y.float()
                optimizer.zero_grad()
                pred = self.model(train_x).float()
                loss = self.loss_function(pred, train_y)
                total_loss += (loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)
                loss.backward()
                optimizer.step()
            mean_loss = total_loss/total_data_num
            print(f"Epoch{e+1} - Mean Loss: {mean_loss}")

class NnTester:
    def __init__(self, model):
        """
        Regressor using trained regression model.

        model: trained regression model
        """
        self.model = model

    def predict(self, pred_loader: DataLoader):
        """
        Regress the input data.

        pred_loader: input data loader to regress
        """
        all_pred_vals = []
        for pred_x, _ in pred_loader:
            pred_x = pred_x.float()
            pred_val = self.model(pred_x).float()
            all_pred_vals.append(pred_val.detach().numpy())
        all_pred_vals = np.vstack(all_pred_vals)
        return all_pred_vals

class RegressorTranPredictor:
    def __init__(self, model_name: str, save_dir: str):
        """
        Regressor for sliding-window data.

        model_name: model name
        save_dir: directory to save model
        """
        self.model_name = model_name
        self.save_dir = save_dir

    def train_dataset(self, train_input: np.ndarray, train_output: np.ndarray, epochs: int, batch: int):
        """
        Train regressor. The trained model is saved.

        train_input: train input data
        train_output: train output (label) data
        epochs: training epoch
        batch: batch size
        """
        input_dim, input_len = train_input.shape[1:]
        output_dim, output_len = train_output.shape[1:]
        trainer = NnTrainer(
            self.model_name, input_dim, input_len, output_dim, output_len
        )
        data_loader = convert_loader(train_input, train_output, batch)
        trainer.fit(data_loader, epochs)
        # torch.save(trainer.model, self.save_dir)
        return trainer.model

    def predict_dataset(self, test_input: list):
        """
        Predict for the input data. Load the trained model.

        test_input: test input data
        """
        model = torch.load(self.save_dir)
        tester = NnTester(model)
        all_preds = []
        for data in test_input:
            data_loader = convert_loader(data, data)
            pred_test = tester.predict(data_loader)
            all_preds.append(pred_test)
        return all_preds
