import os
from datetime import timedelta
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from data.compressor import find_asset_cols, find_interval_set

def _subplot_legends(axs: np.ndarray):
    """
    Legends show of subplots

    axs: subplot axs
    """
    if len(axs.shape) > 1:
        for r in range(axs.shape[0]):
            for c in range(axs.shape[1]):
                axs[r,c].legend()
    else:
        for r in range(axs.shape[0]):
            axs[r].legend()

def _find_high_index(data: np.ndarray):
    """
    Find the points with high values.

    data: data to find
    """
    high_threshold = 2.5*data.mean()
    high_index = np.arange(len(data))[data>high_threshold]
    if len(high_index) > 0:
        return high_index[-1], high_threshold
    else:
        return None, None

class PlotAsset:
    def __init__(self, data_dict: dict, save_path: str):
        """
        Plot all the assets-related information.

        data_dict: dictionary of data
        save_path: path to save figures
        """
        self.data_dict = data_dict
        self.save_path = save_path
        self.default_color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # self._path_create()
        self.running_intervals = find_interval_set(self.data_dict)

    def _path_create(self):
        """
        Create paths to save figures.
        """
        for data_name in self.data_dict.keys():
            if not os.path.exists(f"{self.save_path}{data_name}/"):
                os.makedirs(f"{self.save_path}{data_name}/")

    def running_plot(self):
        """
        Plot the running column of each asset.
        """
        for data_name in self.data_dict.keys():
            dataset = self.data_dict[data_name]
            plt.figure(figsize=(12,9))
            for asset_id, asset in enumerate(dataset.keys()):
                asset_dt = dataset[asset]
                running_col = [c for c in asset_dt.columns if 'running' in c.lower()][0]
                plt.scatter(asset_dt.index, asset_dt[running_col]*asset_id, s=3, label=asset)
            plt.legend()
            plt.savefig(f"{self.save_path}{data_name}/running assets")
            plt.close()

    def press_whole_plot(self):
        """
        Plot the whole columns.
        """
        for data_name in self.data_dict.keys():
            dataset = self.data_dict[data_name]
            for asset in dataset.keys():
                save_dir = f"{self.save_path}{asset}/{data_name}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                asset_dt = dataset[asset]
                dt_intervals = self.running_intervals[data_name][asset]
                for interval_id, interval in enumerate(dt_intervals):
                    plt.figure(figsize=(12,9))
                    for col in asset_dt.columns:
                        if col == "timestamp":
                            continue
                        interval_dt = asset_dt.loc[interval[0]:interval[1]]
                        plt.plot(interval_dt[col], label=col)
                    plt.legend()
                    plt.savefig(f"{save_dir}whole{interval_id}")
                    plt.close()

    def press_temp_plot(self):
        """
        Plot the pressure and temperature columns.
        """
        col_names = ["inlet press", "outlet press", "outlet temp"]
        for data_name in self.data_dict.keys():
            dataset = self.data_dict[data_name]
            for asset in dataset.keys():
                save_dir = f"{self.save_path}{asset}/{data_name}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                asset_dt = dataset[asset]
                dt_intervals = self.running_intervals[data_name][asset]
                plot_cols = find_asset_cols(asset_dt.columns, col_names)
                if "HP" in asset:
                    plot_seconds = 1100
                    val_high = 100
                elif "MP" in asset:
                    plot_seconds = 2200
                    val_high = 130
                for interval_id, interval in enumerate(dt_intervals):
                    interval_dt = asset_dt.loc[interval[0]:interval[1]]
                    plt.figure(figsize=(12,9))
                    for col in plot_cols:
                        plt.plot(interval_dt[col], label=col)
                        plt.xlim((interval_dt.index[0], interval_dt.index[0]+timedelta(seconds=plot_seconds)))
                        plt.ylim((0, val_high))
                    plt.legend()
                    plt.savefig(f"{save_dir}press_temp{interval_id}")
                    plt.close()

    def plot_sliding_std(self, col_names: list, save_name: str):
        """
        Plot the sliding window standard deviation.

        col_names: asset columns to plot
        save_name: save file name
        """
        std_window = 10
        for data_name in self.data_dict.keys():
            dataset = self.data_dict[data_name]
            for asset in dataset.keys():
                asset_dt = dataset[asset]
                dt_intervals = self.running_intervals[data_name][asset]
                plot_cols = find_asset_cols(asset_dt.columns, col_names)
                for interval_id, interval in enumerate(dt_intervals):
                    interval_dt = asset_dt.loc[interval[0]:interval[1]]
                    fig, axs = plt.subplots(2, 2, figsize=(24,14))
                    for col in plot_cols:
                        col_std = sliding_window_view(interval_dt[col], std_window).std(axis=1)
                        high_id, high_val = _find_high_index(col_std)

                        axs[0,0].plot(interval_dt[col], label=col)
                        axs[0,1].plot(interval_dt.index[std_window-1:], col_std, label=col)
                        if high_id is None:
                            continue
                        axs[0,1].axhline(high_val, color='r', linestyle='--')
                        axs[1,0].plot(interval_dt.index[std_window-1:][high_id:], interval_dt[col].iloc[std_window-1:].iloc[high_id:], label=col)
                        axs[1,1].plot(interval_dt.index[std_window-1:][high_id:], col_std[high_id:], label=col)
                    _subplot_legends(axs)
                    plt.savefig(f"{self.save_path}{asset}/{data_name}_{save_name}_sliding_std{interval_id}")
                    plt.close()

    def plot_difference(self, col_names: list, save_name: str):
        """
        Plot the time-difference values.

        col_names: asset columns to plot
        save_name: save file name
        """
        period = 15
        for data_name in self.data_dict.keys():
            dataset = self.data_dict[data_name]
            for asset in dataset.keys():
                asset_dt = dataset[asset]
                dt_intervals = self.running_intervals[data_name][asset]
                plot_cols = find_asset_cols(asset_dt.columns, col_names)
                for interval_id, interval in enumerate(dt_intervals):
                    interval_dt = asset_dt.loc[interval[0]:interval[1]]
                    fig, axs = plt.subplots(2, 1, figsize=(12,14))
                    for col in plot_cols:
                        diff = interval_dt[col].diff(periods=period)
                        axs[0].plot(interval_dt, interval_dt[col], label=col)
                        axs[1].plot(interval_dt.index[period:], diff.iloc[period:], label=col)
                    _subplot_legends(axs)
                    plt.savefig(f"{self.save_path}{asset}/{data_name}_{save_name}_difference{interval_id}")
                    plt.close()

    def asset_length_hist(self, asset_types: list):
        """
        Plot the histogram of interval length of each asset.

        asset_types: asset type list
        """
        type_dataset = {t:[] for t in asset_types}
        for data_name in self.running_intervals.keys():
            for asset in self.running_intervals[data_name].keys():
                for asset_t in type_dataset.keys():
                    if asset_t in asset:
                        for interval in self.running_intervals[data_name][asset]:
                            type_dataset[asset_t].append((interval[1] - interval[0]).total_seconds())
        for asset_t in type_dataset.keys():
            all_perc = []
            for p in range(10, 100, 10):
                perc = round(np.percentile(type_dataset[asset_t], p))
                all_perc.append(perc)
            print(f"{asset_t}: {all_perc}")
            plt.hist(type_dataset[asset_t], bins=20)
            for perc in all_perc:
                plt.axvline(perc, color='r', linestyle='--')
            plt.savefig(f"{self.save_path}{asset_t}_length")
            plt.close()
