import os
from datetime import timedelta
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from data.compressor import DataDivider
from analyzer.autoencode import DivideEncoder
from asset.hyundai.data import split_interval_list, split_interval_dict, remove_asset_part

def _merge_divided_data(data_keys: list, divide_dict: OrderedDict):
    """
    Merge the divided data for each sequence.

    data_keys: list of data key
    divide_dict: dictionary of divided data
    """
    merged_data = OrderedDict()
    for divide_id in divide_dict.keys():
        for dt_id, dt_key in enumerate(data_keys[divide_id]):
            if divide_id == 0:
                merged_data[dt_key] = []
            merged_data[dt_key].append(divide_dict[divide_id][dt_id])
    for dt_key in merged_data.keys():
        stacked_data = np.hstack(merged_data[dt_key]).reshape(-1,1).T
        merged_data[dt_key] = pd.DataFrame(stacked_data, index=[dt_key], columns=range(stacked_data.shape[1]))
    merged_df = pd.concat(merged_data.values())
    return merged_df

class CnnTrainEncode:
    def __init__(self, save_dir: str, asset_type: str, divide_len: int, divide_num: int, normalize: bool):
        """
        1D-CNN based encoder model trainer.

        save_dir: directory to save models
        asset_type: type of asset. (HP or MP)
        divide_len: length of each division
        divide_num: number of division
        normalize: normalize the data or not
        """
        self.save_dir = save_dir
        self.asset_type = asset_type
        self.divide_len = divide_len
        self.divide_num = divide_num
        self.data_divider = DataDivider(divide_len, divide_num, normalize, f"{save_dir}{asset_type}")
        self.divide_encoder = DivideEncoder(divide_num, f"{save_dir}{asset_type}")
    
    def train_models(self, train_data: dict, epochs: int, batch: int):
        """
        train_data: train data dictionary
        epochs: number of epochs
        batch: batch size
        """
        interval_train_data = split_interval_list(train_data, self.divide_len)
        all_train_list = []
        for asset in interval_train_data.keys():
            if self.asset_type not in asset:
                continue
            asset_train_dt = remove_asset_part(interval_train_data[asset])
            all_train_list.extend(asset_train_dt)
        
        print(f"{self.asset_type} train start")
        train_input = self.data_divider.train_compose(all_train_list)
        self.divide_encoder.train_dataset(train_input, epochs, batch)

    def encode_models(self, encode_data: dict):
        """
        Encode the data of dictionary.

        encode_data: dictionary of encode data
        """
        print(f"{self.asset_type} encode start")
        interval_encode_data = split_interval_dict(encode_data, self.divide_len)
        encode_input = OrderedDict()
        for asset in interval_encode_data.keys():
            if self.asset_type not in asset:
                continue
            asset_encode_dt = interval_encode_data[asset]
            for data_name in asset_encode_dt.keys():
                name_encode_dt = asset_encode_dt[data_name]
                key_list = list(name_encode_dt.keys())
                removed_encode_dt = remove_asset_part(name_encode_dt.values())
                for interval_id, interval_dt in enumerate(removed_encode_dt):
                    encode_input[f"{asset},{data_name},{key_list[interval_id]}"] = interval_dt
        encode_key, encode_input = self.data_divider.test_compose(encode_input)
        encode_result = self.divide_encoder.encode_dataset(encode_input)
        encode_result = _merge_divided_data(encode_key, encode_result)
        return encode_result

class EncodeCluster:
    def __init__(self, divide_num: int, save_dir=""):
        """
        Cluster the encoded values of each division.

        divide_num: number of divisions
        save_dir: directory to save figure
        """
        self.model = DBSCAN()
        self.divide_num = divide_num
        self.save_dir = save_dir

    def _scatter_cluster(self, x1, x2, clusters, fig_name):
        """
        Scatter plot the cluster result.

        x1: first coordinate values
        x2: second coordinate values
        clusters: allocated clusters
        fig_name: figure name to save
        """
        unique_cluster = np.unique(clusters)
        for c in unique_cluster:
            cluster_x1 = x1[clusters==c]
            cluster_x2 = x2[clusters==c]
            plt.scatter(cluster_x1, cluster_x2, s=3, label=c)
        plt.legend()
        plt.title(fig_name)
        plt.xlabel("mean")
        plt.ylabel("std")
        plt.savefig(f"{self.save_dir}encode cluster/{fig_name}")
        plt.close()

    def cluster_data(self, data: pd.DataFrame, data_name: str, scatter_plot=False):
        """
        Cluster the given data with mean and standard deviation.

        data: data to cluster
        data_name: name of data to save
        scatter_plot: scatter plot save or not
        """
        all_cluster_result = []
        for divide_id in range(self.divide_num):
            divide_data = data.loc[:,10*divide_id:10*(divide_id+1)-1].dropna(axis=0)
            mean, std = divide_data.mean(axis=1), divide_data.std(axis=1)
            mean_std = np.vstack([mean, std]).T
            mean_std = (mean_std - mean_std.mean(axis=0)) / mean_std.std(axis=0)
            divide_clusters = self.model.fit_predict(mean_std)
            if scatter_plot:
                self._scatter_cluster(mean, std, divide_clusters, f"{data_name}_divide{divide_id}")
            all_cluster_result.append(pd.DataFrame(divide_clusters, index=divide_data.index, columns=[divide_id]))
        all_cluster_result = pd.concat(all_cluster_result, axis=1)
        return all_cluster_result

class EncodeAnalyze:
    def __init__(self, encode_data: dict, divide_len: int, divide_num: int, plot_dir: str):
        """
        encode_data: sequential data dictionary to encode
        divide_len: length of division
        divide_num: number of divisions
        plot_dir: directory to save plots
        """
        self.encode_data = split_interval_dict(encode_data, divide_len)
        self.divide_len = divide_len
        self.divide_num = divide_num
        self.plot_dir = plot_dir

    def encode_data_result(self, encode_result: pd.DataFrame):
        """
        Plot the results of encode.

        encode_result: result of encoding
        """
        y_low = np.nanpercentile(encode_result, 0.5)
        y_high = np.nanpercentile(encode_result, 99.5)
        max_abs = max(abs(y_low), y_high)
        for dt_key in encode_result.index:
            asset, dt_name, interval_id = dt_key.split(",")
            asset_plot_dir = f"{self.plot_dir}{asset}/"
            if not os.path.exists(asset_plot_dir):
                os.makedirs(asset_plot_dir)
            encode_input = self.encode_data[asset][dt_name][int(interval_id)]
            encode_input = encode_input.loc[:,encode_input.std(axis=0)>0]
            encode_val = encode_result.loc[dt_key]
            if "HP" in asset:
                val_high = 100
            elif "MP" in asset:
                val_high = 130
            fig, axs = plt.subplots(2,1, figsize=(10,7))
            for col in encode_input.columns:
                axs[0].plot(encode_input[col], label=col)
            axs[0].legend()
            axs[0].set_xlim((encode_input.index[0], encode_input.index[0]+timedelta(seconds=self.divide_len*self.divide_num+1)))
            for d_id in range(1, self.divide_num):
                axs[0].axvline(encode_input.index[0]+timedelta(seconds=self.divide_len*d_id), color='r', linestyle='--')
            axs[0].set_ylim((0, val_high))
            axs[1].plot(encode_val)
            axs[1].set_ylim((-max_abs, max_abs))
            axs[1].set_xlim((-1, 10*self.divide_num+1))
            for d_id in range(1, self.divide_num):
                axs[1].axvline(10*d_id, color='r', linestyle='--')
            fig.suptitle(f"{dt_name} - {asset} - interval{interval_id}")
            plt.savefig(f"{asset_plot_dir}{dt_name}_interval{interval_id}")
            plt.close()

    def cluster_data_result(self, cluster_result: pd.DataFrame):
        """
        Plot the cluster result with sequential data. If the cluster is -1, the interval (division) is painted red.

        cluster_result: cluster of each division of each sequential data.
        """
        plot_dir = f"{self.plot_dir}cluster result/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for dt_key in cluster_result.index:
            asset, dt_name, interval_id = dt_key.split(",")
            encode_input = self.encode_data[asset][dt_name][int(interval_id)]
            encode_input = encode_input.loc[:,encode_input.std(axis=0)>0]
            interval_clusters = cluster_result.loc[dt_key]
            if "HP" in asset:
                val_high = 100
            elif "MP" in asset:
                val_high = 130
            for col in encode_input.columns:
                plt.plot(encode_input[col], label=col)
            plt.legend()
            plt.xlim((encode_input.index[0], encode_input.index[0]+timedelta(seconds=self.divide_len*self.divide_num+1)))
            plt.ylim((0, val_high))
            for d_id in range(self.divide_num):
                plt.axvline(encode_input.index[0]+timedelta(seconds=self.divide_len*(d_id+1)), color='r', linestyle='--')
                if interval_clusters[d_id] == -1:
                    plt.axvspan(encode_input.index[0]+timedelta(seconds=self.divide_len*d_id), encode_input.index[0]+timedelta(seconds=self.divide_len*(d_id+1)),
                                color='r', alpha=0.3)
            plt.title(f"{dt_name} - {asset} - interval{interval_id}")
            plt.savefig(f"{plot_dir}{asset}_{dt_name}_interval{interval_id}")
            plt.close()
