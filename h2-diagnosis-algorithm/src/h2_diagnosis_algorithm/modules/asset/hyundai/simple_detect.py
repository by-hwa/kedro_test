import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from analyzer.simple_algorithm import SimpleAlgorithm
from asset.hyundai.data import split_interval_list, split_interval_dict, list_to_dict

class SimpleTrainTest:
    def __init__(self, save_dir: str, std_len:int, min_len: int):
        self.save_dir = save_dir
        self.min_len = min_len
        self.algorithm = SimpleAlgorithm(std_len)

    def train_models(self, train_data: dict):
        interval_train_data = split_interval_list(train_data, self.min_len, True)
        for asset in interval_train_data.keys():
            save_dir = f"{self.save_dir}simple/{asset}"
            asset_train_dt = interval_train_data[asset]
            self.algorithm.train(asset_train_dt, save_dir)

    def test_models(self, test_data: dict):
        interval_test_data = split_interval_dict(test_data, self.min_len, False)
        all_data, all_results = OrderedDict(), OrderedDict()
        for asset in interval_test_data.keys():
            save_dir = f"{self.save_dir}simple/{asset}.csv"
            if not os.path.exists(save_dir):
                print(f"{asset} model not found")
                continue
            asset_test_dt = interval_test_data[asset]
            asset_test_list, asset_test_keys = [], {}
            for data_name in asset_test_dt.keys():
                asset_test_keys[data_name] = list(asset_test_dt[data_name].keys())
                for interval_id in asset_test_dt[data_name].keys():
                    asset_test_list.append(asset_test_dt[data_name][interval_id])
            test_results = self.algorithm.test(asset_test_list, f"{save_dir}")
            all_data[asset] = list_to_dict(asset_test_list, asset_test_keys)
            all_results[asset] = list_to_dict(test_results, asset_test_keys)
        return all_data, all_results

class ResultJudge:
    def __init__(self, standard_num: int):
        self.standard_num = standard_num

    def plot_data_error(self, true_val, result_val, save_dir: str):
        for asset in true_val.keys():
            asset_trues = true_val[asset]
            asset_result = result_val[asset]
            for dt_name in asset_trues.keys():
                plot_save_dir = f"{save_dir}{asset}/{dt_name}/"
                if not os.path.exists(plot_save_dir):
                    os.makedirs(plot_save_dir)
                split_trues = asset_trues[dt_name]
                split_results = asset_result[dt_name]
                for interval_id in split_trues.keys():
                    interval_true = split_trues[interval_id]
                    interval_result = split_results[interval_id]
                    for col in interval_result.columns:
                        fig, axs = plt.subplots(2,1, figsize=(10,7))
                        axs[0].plot(interval_true[col])
                        axs[1].plot(interval_result[col])
                        if interval_result[col].sum() > self.standard_num:
                            axs[1].axvspan(interval_result.index[0], interval_result.index[-1], color='r', alpha=0.3)
                        fig.suptitle(f"{dt_name} - {asset} - interval{interval_id} - {col}")
                        plt.savefig(f"{plot_save_dir}{col}_interval{interval_id}")
                        plt.close()
