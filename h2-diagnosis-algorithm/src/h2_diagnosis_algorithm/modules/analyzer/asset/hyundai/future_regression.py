import os
from collections import OrderedDict
from itertools import groupby
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
from data.compressor import DataSlider, ErrorAnalyzer
from analyzer.regression import RegressorTranPredictor
from asset.hyundai.data import split_interval_list, split_interval_dict, list_to_dict, read_variable_cols, reform_window_data, remove_asset_part

def _asset_type_allocate(data_dict: dict, asset_type: str):
    """
    Allocate the asset type of each data.

    data_dict: dictionary of data
    asset_type: types of assets
    """
    type_data = OrderedDict()
    for data_name in data_dict.keys():
        type_data[data_name] = OrderedDict()
        for asset in data_dict[data_name].keys():
            if asset_type in asset:
                type_data[data_name][asset] = data_dict[data_name][asset]
    return type_data

def _get_cluster_interval(clusters: pd.DataFrame):
    """
    Get the continuous intervals of cluster id >= 0.

    clusters: cluster data
    """
    in_clusters = clusters.index[clusters>=0]
    all_intervals = []
    for _, g in groupby(enumerate(in_clusters), lambda ix : ix[0] - ix[1]):
        interval_id = list(map(itemgetter(1), g))
        all_intervals.append([interval_id[0], interval_id[-1]])
    return all_intervals

def _divide_cluster_data(data: pd.DataFrame, cluster: pd.DataFrame, divide_len: int):
    """
    Divide the sequence data by cluster id. Continuous interval with cluster id >= 0 is extracted as one sequence.

    data: sequence dataset
    cluster: cluster data
    divide_len: length of division
    """
    cluster_intervals = _get_cluster_interval(cluster)
    all_divide_data = []
    for interval in cluster_intervals:
        start, end = interval[0], interval[1]
        interval_data = data.iloc[divide_len*start:divide_len*(end+1)]
        all_divide_data.append(interval_data)
    return all_divide_data



class CnnTrainTest:
    def __init__(self, model_name: str, save_dir: str, divide_len: dict, asset_type: str, input_len: int, output_len: int, normalize: bool, divide: bool):
        """
        Train and test the 1D-CNN regression model.

        model_name: model name
        save_dir: directory to save models
        divide_len: length of division
        asset_type: asset types
        input_len: length of input window
        output_len: length of label window
        normalize: normalize or not
        divide: divide the data or not
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.divide_len = divide_len
        self.asset_type = asset_type
        self.input_len = input_len
        self.output_len = output_len
        self.data_slider = DataSlider(input_len, output_len, normalize)
        if divide:
            model_save_dir = f"{save_dir}{model_name}_divide/{asset_type}"
        else:
            model_save_dir = f"{save_dir}{model_name}/{asset_type}"
        self.train_predictor = RegressorTranPredictor(model_name, model_save_dir)
    
    def train_asset(self, train_data: dict, epochs: int, batch: int):
        """
        Train models of assets.

        train_data: trian data dictionary
        epochs: epoch number
        batch: batch size
        """
        asset_train = _asset_type_allocate(train_data, self.asset_type)
        interval_data = split_interval_list(asset_train, self.divide_len[self.asset_type] + self.input_len + self.output_len)

        all_asset_data = []
        for asset in interval_data.keys():
            asset_dt = interval_data[asset]
            for interval_dt in asset_dt:
                all_asset_data.append(interval_dt.iloc[self.divide_len[self.asset_type]:])
        print(f"{self.asset_type} train start")
        save_dir = f"{self.save_dir}{self.model_name}/{self.asset_type}"
        asset_train_dt = remove_asset_part(all_asset_data)
        train_input, train_output = self.data_slider.train_compose(asset_train_dt, f"{save_dir}.csv")
        return self.train_predictor.train_dataset(train_input, train_output, epochs, batch)

    def test_models(self, test_data: dict):
        """
        Test models of assets.

        test_data: test data dictionary
        """
        asset_test = _asset_type_allocate(test_data, self.asset_type)
        interval_data = split_interval_dict(asset_test, self.divide_len[self.asset_type] + self.input_len + self.output_len)
        
        all_pred, all_output, all_times = OrderedDict(), OrderedDict(), OrderedDict()
        for asset in interval_data.keys():
            print(f"{asset} test start")
            save_dir = f"{self.save_dir}{self.model_name}/{self.asset_type}"
            asset_test_dt = interval_data[asset]
            asset_test_list, asset_test_keys = [], OrderedDict()
            for data_name in asset_test_dt.keys():
                asset_test_keys[data_name] = list(asset_test_dt[data_name].keys())
                for interval_id in asset_test_dt[data_name].keys():
                    interval_dt = asset_test_dt[data_name][interval_id]
                    asset_test_list.append(interval_dt.iloc[self.divide_len[self.asset_type]:])
            asset_test_list = remove_asset_part(asset_test_list)
            test_input, test_output, test_times = self.data_slider.test_compose(asset_test_list, f"{save_dir}.csv")
            test_pred = self.train_predictor.predict_dataset(test_input)
            all_pred[asset] = list_to_dict(test_pred, asset_test_keys)
            all_output[asset] = list_to_dict(test_output, asset_test_keys)
            all_times[asset] = list_to_dict(test_times, asset_test_keys)
        return all_pred, all_output, all_times
    
    def divdie_train_asset(self, train_data: dict, cluster_result: pd.DataFrame, epochs: int, batch: int):
        """
        Divide the train data and train model of each division.

        train_data: train data dictionary
        cluster_result: cluster data
        epochs: epoch number
        batch: batch size
        """
        asset_train = _asset_type_allocate(train_data, self.asset_type)
        interval_data = split_interval_dict(asset_train, self.divide_len[self.asset_type] + self.input_len + self.output_len)
        
        all_asset_data = []
        for asset in interval_data.keys():
            asset_dt = interval_data[asset]
            for data_name in asset_dt.keys():
                name_dt = asset_dt[data_name]
                for interval_id in name_dt.keys():
                    interval_dt = name_dt[interval_id]
                    interval_cluster = cluster_result.loc[f"{asset},{data_name},{interval_id}"]
                    divide_interval_dt = _divide_cluster_data(interval_dt, interval_cluster[1:], self.divide_len[self.asset_type])
                    all_asset_data.extend(divide_interval_dt)
        print(f"{self.asset_type} train start")
        save_dir = f"{self.save_dir}{self.model_name}_divide/{self.asset_type}"
        asset_train_dt = remove_asset_part(all_asset_data)
        train_input, train_output = self.data_slider.train_compose(asset_train_dt, f"{save_dir}.csv")
        self.train_predictor.train_dataset(train_input, train_output, epochs, batch)
    
    def divdie_test_models(self, test_data: dict):
        """
        Divide the test data and test model of each division.

        test_data: test data dictionary
        """
        asset_test = _asset_type_allocate(test_data, self.asset_type)
        interval_data = split_interval_dict(asset_test, self.divide_len[self.asset_type] + self.input_len + self.output_len)

        all_pred, all_output, all_times = OrderedDict(), OrderedDict(), OrderedDict()
        for asset in interval_data.keys():
            print(f"{asset} test start")
            save_dir = f"{self.save_dir}{self.model_name}_divide/{self.asset_type}"
            asset_test_dt = interval_data[asset]
            asset_test_list, asset_test_keys = [], OrderedDict()
            for data_name in asset_test_dt.keys():
                asset_test_keys[data_name] = list(asset_test_dt[data_name].keys())
                for interval_id in asset_test_dt[data_name].keys():
                    interval_dt = asset_test_dt[data_name][interval_id]
                    asset_test_list.append(interval_dt.iloc[self.divide_len[self.asset_type]:])
            asset_test_list = remove_asset_part(asset_test_list)
            test_input, test_output, test_times = self.data_slider.test_compose(asset_test_list, f"{save_dir}.csv")
            test_pred = self.train_predictor.predict_dataset(test_input)
            all_pred[asset] = list_to_dict(test_pred, asset_test_keys)
            all_output[asset] = list_to_dict(test_output, asset_test_keys)
            all_times[asset] = list_to_dict(test_times, asset_test_keys)
        return all_pred, all_output, all_times

class ErrorJudge:
    def __init__(self, train_true: dict, train_pred: dict, asset_type: str, variable_dir: str, threshold: float, standard_num: int):
        """
        Judge and detect anomaly with the regression error.

        train_true: true value of train data
        train_pred: pred value of train data
        asset_type: asset type
        variable_dir: directory to variable columns
        threshold: judge threshold
        standard_num: standard number to judge anomaly 
        """
        self.variable_cols = read_variable_cols(variable_dir)
        self.asset_type = asset_type
        self.threshold = threshold
        self.standard_num = standard_num
        self.error_analyzer = ErrorAnalyzer(3)
        self.error_standards = self._compute_error_standard(train_true, train_pred)

    def _compute_error_standard(self, train_true: dict, train_pred: dict):
        """
        Compute the high and low standard to judge the error.

        train_true: true value of train data
        train_pred: pred value of train data
        """
        all_asset_err = []
        for asset in train_true.keys():
            asset_trues = train_true[asset]
            asset_preds = train_pred[asset]
            for dt_name in asset_preds.keys():
                split_trues = asset_trues[dt_name]
                split_preds = asset_preds[dt_name]
                for interval_id in split_trues.keys():
                    interval_true = split_trues[interval_id]
                    interval_pred = split_preds[interval_id]
                    window_error = interval_true - interval_pred
                    window_error = reform_window_data(window_error, self.variable_cols[self.asset_type])
                    all_asset_err.append(window_error)
        asset_err = pd.concat(all_asset_err, ignore_index=True)
        low_standard, high_standard = self.error_analyzer.compute_error_range(asset_err)
        all_standards = {"low": low_standard, "high": high_standard}
        return all_standards
    
    def compute_test_error(self, true_val: dict, pred_val: dict):
        """
        Compute the test data error.

        true_val: true value
        pred_val: pred value
        """
        all_test_err = {}
        for asset in true_val.keys():
            asset_trues = true_val[asset]
            asset_preds = pred_val[asset]
            all_test_err[asset] = {}
            for dt_name in asset_preds.keys():
                split_trues = asset_trues[dt_name]
                split_preds = asset_preds[dt_name]
                all_test_err[asset][dt_name] = {}
                for interval_id in split_trues.keys():
                    interval_true = split_trues[interval_id]
                    interval_pred = split_preds[interval_id]
                    interval_error = interval_true - interval_pred
                    all_test_err[asset][dt_name][interval_id] = interval_error
        return all_test_err
    
    def compute_test_results(self, test_errors: dict, test_times: dict):
        """
        Compute the results of judgement for test dataset.

        test_errors: test data error
        test_times: test data time
        """
        all_outlie_rate, all_judge = {}, {}
        for asset in test_errors.keys():
            asset_errors = test_errors[asset]
            asset_times = test_times[asset]
            all_outlie_rate[asset], all_judge[asset] = {}, {}
            for dt_name in asset_errors.keys():
                split_errors = asset_errors[dt_name]
                split_times = asset_times[dt_name]
                all_outlie_rate[asset][dt_name] = {}
                all_judge[asset][dt_name] = {}
                for interval_id in split_errors.keys():
                    interval_error = split_errors[interval_id]
                    interval_time = split_times[interval_id].flatten()
                    interval_error = reform_window_data(interval_error, self.variable_cols[self.asset_type], interval_time)
                    columns_outlie = (interval_error < self.error_standards["low"]) | (interval_error > self.error_standards["high"])
                    outlie_rate = columns_outlie.sum(axis=1) / columns_outlie.shape[1]
                    all_outlie_rate[asset][dt_name][interval_id] = outlie_rate
                    all_judge[asset][dt_name][interval_id] = (outlie_rate>self.threshold).sum() > self.standard_num
        return all_outlie_rate, all_judge
    
    def plot_data_error(self, true_val: dict, pred_val: dict, errors: dict, times: dict, save_dir: str):
        """
        Plot the error of prediction value.

        true_val: true value
        pred_val: prediction value
        errors: error values
        times: time of true value
        save_dir directory to save figures
        """
        for asset in true_val.keys():
            asset_trues = true_val[asset]
            asset_preds = pred_val[asset]
            asset_errors = errors[asset]
            asset_times = times[asset]
            for dt_name in asset_preds.keys():
                plot_save_dir = f"{save_dir}{asset}/{dt_name}/"
                if not os.path.exists(plot_save_dir):
                    os.makedirs(plot_save_dir)
                split_trues = asset_trues[dt_name]
                split_preds = asset_preds[dt_name]
                split_errors = asset_errors[dt_name]
                split_times = asset_times[dt_name]
                for interval_id in split_trues.keys():
                    interval_time = split_times[interval_id].flatten()
                    interval_true = reform_window_data(split_trues[interval_id], self.variable_cols[self.asset_type], interval_time)
                    interval_pred = reform_window_data(split_preds[interval_id], self.variable_cols[self.asset_type], interval_time)
                    interval_error = reform_window_data(split_errors[interval_id], self.variable_cols[self.asset_type], interval_time)
                    for col in interval_true.columns:
                        fig, axs = plt.subplots(2,1, figsize=(10,7))
                        axs[0].plot(interval_true[col], alpha=0.4, label="true value")
                        axs[0].plot(interval_pred[col], alpha=0.4, label="prediction")
                        axs[0].legend()
                        axs[1].plot(interval_error[col])
                        axs[1].axhline(self.error_standards["low"][col], color='r', linestyle='--')
                        axs[1].axhline(self.error_standards["high"][col], color='r', linestyle='--')
                        fig.suptitle(f"{dt_name} - {asset} - interval{interval_id} - {col}")
                        plt.savefig(f"{plot_save_dir}{col}_interval{interval_id}")
                        plt.close()

    def plot_results(self, outlie_rate: dict, judge: dict, save_dir: str):
        """
        Plot the anomaly detection result with sequential data. If the division is anomaly, the interval (division) is painted red.

        outlie_rate: the number rate of lying out of threshold
        judge: anomaly detection result
        save_dir: directory to save figures
        """
        for asset in outlie_rate.keys():
            asset_rate = outlie_rate[asset]
            asset_judge = judge[asset]
            for dt_name in asset_rate.keys():
                plot_save_dir = f"{save_dir}{asset}/{dt_name}/"
                if not os.path.exists(plot_save_dir):
                    os.makedirs(plot_save_dir)
                split_rate = asset_rate[dt_name]
                split_judge = asset_judge[dt_name]
                for interval_id in split_rate.keys():
                    interval_rate = split_rate[interval_id]
                    interval_judge = split_judge[interval_id]
                    plt.plot(interval_rate)
                    plt.ylim(-0.1, 1.1)
                    if interval_judge:
                        plt.axvspan(interval_rate.index[0], interval_rate.index[-1], color='r', alpha=0.3)
                    plt.savefig(f"{plot_save_dir}out_rate_interval{interval_id}")
                    plt.close()
