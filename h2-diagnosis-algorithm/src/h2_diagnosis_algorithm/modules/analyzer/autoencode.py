import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

def convert_loader(input_data: np.ndarray, batch=32) -> DataLoader:
    """
    Convert input data (np.ndarray) to torch DataLoader.

    input_data: Train or predict input data feeding to torch model.
    batch: batch size.
    """
    dt_loader = DataLoader(input_data, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

class AutoencoderCnn1d(nn.Module):
    def __init__(self, input_dim: int, input_len: int):
        """
        1D-CNN based Autoencoder model.

        input_dim: dimension (channel size) of input data
        input_len: length (window size) of input data
        """
        super(AutoencoderCnn1d, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encode_len = 10
        self._define_encoder()
        self._define_decoder()

    def _define_encoder(self):
        """
        Define encoder layers. 
        Last layer's kernel size and stride is set to be encode_kernel in order to make latent vector size as encode length. 
        """
        encode_kernel = self.input_len // self.encode_len
        self.encode_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=4, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=encode_kernel, stride=encode_kernel)

    def _define_decoder(self):
        """
        Define decoder layers.
        First layer is defined to be FNN to upsize to input length.
        """
        self.decode_layer1 = nn.Linear(self.encode_len, self.input_len)
        self.decode_layer2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=4, out_channels=self.input_dim, kernel_size=5, padding="same")

    def encode(self, x):
        """
        Encode the input data.

        x: input data
        """
        x = torch.relu(self.encode_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = self.encode_layer3(x)
        x = x.view(-1, self.encode_len)
        return x
    
    def decode(self, x):
        """
        Decode the input data.

        x: input data (latent vector)
        """
        x = torch.relu(self.decode_layer1(x))
        x = x.view(-1, 1, self.input_len)
        x = torch.relu(self.decode_layer2(x))
        x = self.decode_layer3(x)
        return x

    def forward(self, x):
        """
        Reconstruct the input data.

        x: input data
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

class Trainer:
    def __init__(self, input_dim: int, input_len: int):
        """
        Autoencoder Trainer.

        input_dim: dimension (channel size) of input data
        input_len: length (window size) of input data
        """
        self.model = AutoencoderCnn1d(input_dim, input_len)    
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        """
        Fit (train) autoencoder. Use NAdam optimizer.

        train_loader: train data loader
        epoch: train epoch
        """
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_loss = 0
            for train_x in train_loader:
                train_x = train_x.float()
                optimizer.zero_grad()
                pred = self.model(train_x).float()
                encode_val = self.model.encode(train_x).float()
                loss = self.loss_function(pred, train_x) + (encode_val**2).mean()
                total_loss += (loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)
                loss.backward()
                optimizer.step()
            mean_loss = total_loss/total_data_num
            print(f"Epoch{e+1} - Mean Loss: {mean_loss}")

class Encoder:
    def __init__(self, model):
        """
        Encoder using trained autoencoder model.

        model: trained autoencoder model
        """
        self.model = model

    def encode(self, dt_loader: DataLoader) -> np.ndarray:
        """
        Encode the input data.

        dt_loader: input data loader to encode
        """
        all_encode_vals = []
        for x in dt_loader:
            x = x.float()
            x_encode = self.model.encode(x).float()
            all_encode_vals.append(x_encode.detach().numpy())
        all_encode_vals = np.vstack(all_encode_vals)
        return all_encode_vals

class DivideEncoder:
    def __init__(self, divide_num: int, save_dir: str):
        """
        Encoder for divided sequential data.

        divide_num: number of division
        save_dir: directory to save model
        """
        self.divide_num = divide_num
        self.save_dir = save_dir

    def train_dataset(self, train_input_dict: dict, epochs: int, batch: int):
        """
        Train autoencoder for each division. The trained model is saved.

        train_input_dict: train input data dictionary. {divide id: input data}
        epochs: training epoch
        batch: batch size
        """
        for divide_id in range(self.divide_num):
            print(f"divide{divide_id} Training")
            train_input = train_input_dict[divide_id]
            input_dim, input_len = train_input.shape[1:]
            data_loader = convert_loader(train_input, batch)
            trainer = Trainer(input_dim, input_len)
            trainer.fit(data_loader, (epochs*len(train_input_dict))//(len(train_input_dict)-divide_id))
            # torch.save(trainer.model, f"{self.save_dir}_divide{divide_id}")
            return trainer.model

    def encode_dataset(self, encode_input_dict: dict, model):
        """
        Encode data for each division. Load the trained model.

        encode_input_dict: encode input data dictionary. {divide id: input data}
        """
        all_encodes = OrderedDict()
        for divide_id in range(self.divide_num):
            # model = torch.load(f"{self.save_dir}_divide{divide_id}")
            encoder = Encoder(model)
            encode_input = encode_input_dict[divide_id]
            data_loader = convert_loader(encode_input, 128)
            all_encodes[divide_id] = encoder.encode(data_loader)
        return all_encodes
    
class EncodeCluster:
    def __init__(self, divide_num: int):
        """
        Encoded value cluster module. DBSCAN is used as the cluster model.

        divide_num: number of division
        """
        self.cluster_model = DBSCAN()
        self.divide_num = divide_num
        self.encode_len = 10

    def cluster_encode(self, data: pd.DataFrame):
        """
        Cluster the mean, std of encoded values of each division.

        data: encoded values of whole sequence
        """
        all_cluster_result = []
        for divide_id in range(self.divide_num):
            divide_data = data.loc[:,self.encode_len*divide_id:self.encode_len*(divide_id+1)-1].dropna(axis=0)
            mean, std = divide_data.mean(axis=1), divide_data.std(axis=1)
            mean_std = np.vstack([mean, std]).T
            mean_std = (mean_std - mean_std.mean(axis=0)) / mean_std.std(axis=0)
            divide_clusters = self.cluster_model.fit_predict(mean_std)
            all_cluster_result.append(pd.DataFrame(divide_clusters, index=divide_data.index, columns=[divide_id]))
        all_cluster_result = pd.concat(all_cluster_result, axis=1)
        return all_cluster_result
