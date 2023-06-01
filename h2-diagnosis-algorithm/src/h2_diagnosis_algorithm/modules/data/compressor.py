from collections import OrderedDict
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from datetime import timedelta
from modules.data.process import find_intervals, find_asset_cols

# def _combine_dict_lists(dict_list: dict):
#     dict_vals = dict_list.values()
#     combined_list = []
#     for l in dict_vals:
#         combined_list.extend(l)
#     return combined_list

# def _large_interval_find(whole_data, large_threshold, small_threshold):
#     period = timedelta(seconds=10)
#     large_index = whole_data.index[whole_data>large_threshold]
#     rest_large_index = list(large_index)
#     all_intervals = []
#     while len(rest_large_index) > 0:
#         start_index = rest_large_index[0]
#         present_interval = [start_index, None]
#         while True:
#             max_period = whole_data.loc[start_index:start_index+period].max()
#             if max_period > small_threshold:
#                 start_index += period
#             else:
#                 present_interval[-1] = start_index
#                 break
#         rest_large_index = [i for i in rest_large_index if i > present_interval[-1]]
#         all_intervals.append(present_interval)
#     return all_intervals

# def _find_start_point(col_data, threshold):
#     large_threshold = threshold[1]
#     small_threshold = threshold[0]
#     if (col_data > large_threshold).sum() < 1:
#          large_threshold = np.percentile(col_data, 95)
#          small_threshold = np.percentile(col_data, 90) #(large_threshold + np.mean(col_data)) / 2
#     large_intervals = _large_interval_find(col_data, large_threshold, small_threshold)
#     small_start_index = col_data.index[col_data.index <= large_intervals[0][-1]][-1]
#     return small_start_index

# def _extract_interval_data(data_dict, all_cols=[]):
#     data_intervals = find_interval_set(data_dict)
#     interval_values = {}
#     for data_name in data_dict.keys():
#         dataset = data_dict[data_name]
#         interval_values[data_name] = {}
#         for asset in dataset.keys():
#             asset_dt = dataset[asset]
#             dt_intervals = data_intervals[data_name][asset]
#             if len(all_cols) > 0:
#                 asset_cols = find_asset_cols(asset_dt.columns, all_cols)
#             else:
#                 asset_cols = asset_dt.columns
#             interval_values[data_name][asset] = {}
#             for interval_id, interval in enumerate(dt_intervals):
#                 interval_cols = [c for c in asset_cols if c in asset_dt.columns]
#                 interval_dt = asset_dt.loc[interval[0]:interval[1], interval_cols]
#                 interval_values[data_name][asset][interval_id] = interval_dt
#     return interval_values

def find_interval_set(data_dict):
    all_intervals = {}
    for data_name in data_dict.keys():
        all_intervals[data_name] = {}
        dataset = data_dict[data_name]
        for asset in dataset.keys():
            asset_dt = dataset[asset]
            dt_intervals = find_intervals(asset_dt.index)
            all_intervals[data_name][asset] = dt_intervals
    return all_intervals

def _find_variable_cols(train_interval_list):
    train_combined_dt = pd.concat(train_interval_list)
    variable_columns = list(train_combined_dt.columns[train_combined_dt.std(axis=0) > 0])
    return variable_columns

def _remove_constant_cols(interval_data_list: list, variable_cols: list):
    variable_data = []
    for data in interval_data_list:
        variable_data.append(data[variable_cols])
    return variable_data

def _compute_statistics(train_interval_list):
    train_combined_dt = pd.concat(train_interval_list)
    return train_combined_dt.mean(axis=0), train_combined_dt.std(axis=0)

def _normalize_data(interval_data_list: list, mean, std):
    all_normalized_data = []
    for data in interval_data_list:
        normalized_data = (data - mean) / std
        all_normalized_data.append(normalized_data)
    return all_normalized_data

def _slide_data(data, input_len, output_len, stride):
    slide_input, slide_output = [], []
    for col in data.columns:
        col_dt = data[col]
        input_dt = sliding_window_view(col_dt.iloc[:-output_len], input_len)
        output_dt = sliding_window_view(col_dt.iloc[input_len:], output_len)
        input_dt = input_dt[range(0, len(input_dt), stride)]
        output_dt = output_dt[range(0, len(output_dt), stride)]
        slide_input.append(input_dt)
        slide_output.append(output_dt)
    slide_input = np.stack(slide_input, 1)
    slide_output = np.stack(slide_output, 1)
    output_times = sliding_window_view(data.index[input_len:], output_len)
    output_times = output_times[range(0, len(output_times), stride)]
    return slide_input, slide_output, output_times

# class TargetExtractor:
#     def __init__(self, train_data, test_data, type_cols: dict, diff_period: int):
#         """
#         Extract the diagnosis target part of each data, based on the difference in a period.
#         The value radically increases in the initial time and the increase-stopping point is captured as the target starting point.
#         This also splits the train-test dataset and extracts columns based on the type columns.

#         data_dict: dictionary of running interval data
#         train_test_name: train, test dataset name
#         type_cols: type of columns to extract
#         diff_period: period (time difference) to compute difference
#         """
#         self.train_data = train_data
#         self.test_data = test_data
#         self.diff_period = diff_period
#         self.all_cols = _combine_dict_lists(type_cols)

#         self.train_intervals = _extract_interval_data(self.train_data, self.all_cols)
#         self.test_intervals = _extract_interval_data(self.test_data, self.all_cols)
#         self.target_threshold = self._extract_thresholds()
#         self.all_train_targets = self._extract_all_targets(self.train_intervals)
#         self.all_test_targets = self._extract_all_targets(self.test_intervals)

#     def extract_train_diff_values(self):
#         """
#         Extract and stack the difference values of train dataset for each column.
#         """
#         train_values = {}
#         for data_name in self.train_intervals.keys():
#             dataset = self.train_intervals[data_name]
#             for asset in dataset.keys():
#                 dt_intervals = dataset[asset]
#                 for interval_id in dt_intervals.keys():
#                     interval_dt = dt_intervals[interval_id]
#                     for col in interval_dt.columns:
#                         if col not in train_values.keys():
#                             train_values[col] = []
#                         train_values[col].append(interval_dt[col].diff(self.diff_period).iloc[self.diff_period:])
#         for col in train_values.keys():
#             train_values[col] = np.concatenate(train_values[col])
#         return train_values
    
#     def _extract_thresholds(self):
#         """
#         Extract difference thresholds of each column, from train dataset.
#         """
#         train_values = self.extract_train_diff_values()
#         thresholds = {}
#         for val_name in train_values.keys():
#             values = train_values[val_name]
#             max_val = np.max(values)
#             mean_val = np.mean(values)
#             high_threshold = (max_val+mean_val)/2
#             low_threshold = np.percentile(values[values<high_threshold], 99)
#             thresholds[val_name] = (low_threshold, high_threshold)
#         return thresholds

#     def _extract_interval_target(self, interval_data: pd.DataFrame):
#         diff_data = interval_data.diff(periods=self.diff_period).iloc[self.diff_period:]
#         all_col_target = {}
#         for col in self.target_threshold.keys():
#             if col not in diff_data.columns:
#                 continue
#             col_dt = diff_data[col]
#             col_threshold = self.target_threshold[col]
#             target_start = _find_start_point(col_dt, col_threshold)
#             col_target = interval_data.loc[target_start:, col]
#             all_col_target[col] = col_target
#         return all_col_target
    
#     def _extract_all_targets(self, interval_data_dict):
#         all_targets = {}
#         for data_name in interval_data_dict.keys():
#             dataset = interval_data_dict[data_name]
#             all_targets[data_name] = {}
#             for asset in dataset.keys():
#                 dt_intervals = dataset[asset]
#                 all_targets[data_name][asset] = {}
#                 for interval_id in dt_intervals.keys():
#                     interval_dt = dt_intervals[interval_id]
#                     target_dt = self._extract_interval_target(interval_dt)
#                     all_targets[data_name][asset][interval_id] = target_dt
#         return all_targets

class DataDivider:
    def __init__(self, divide_len: int, divide_num: int, normalize: bool, save_dir_name: str):
        """
        Sequential data divider.

        divide_len: division length
        divide_num: number of division
        normalize: normalize or not (boolean)
        save_dir_name: directory/file name to save the mean, std and variable columns
        """
        self.divide_len = divide_len
        self.divide_num = divide_num
        self.normalize = normalize
        self.save_dir_name = save_dir_name
        self.statistic = dict()

    def _divide_dataset_dict(self, data_dict: OrderedDict):
        """
        Divdie each data of data dictionary.

        data_dict: dictionary of dataset
        """
        interval_data_list = {i:[] for i in range(self.divide_num)}
        interval_data_key = {i:[] for i in range(self.divide_num)}
        for dt_key in data_dict.keys():
            dt = data_dict[dt_key]
            max_divide = len(dt) // self.divide_len
            for i in range(min(max_divide, self.divide_num)):
                interval_data_list[i].append(dt.iloc[self.divide_len*i:self.divide_len*(i+1)])
                interval_data_key[i].append(dt_key)
        return interval_data_key, interval_data_list

    def _divide_dataset_list(self, data_list: list):
        """
        Divdie each data of data list.

        data_list: list of dataset
        """
        interval_data_list = {i:[] for i in range(self.divide_num)}
        for dt in data_list:
            max_divide = len(dt) // self.divide_len
            for i in range(min(max_divide, self.divide_num)):
                interval_data_list[i].append(dt.iloc[self.divide_len*i:self.divide_len*(i+1)])
        return interval_data_list
    
    def _stack_data(self, interval_data_dict: dict):
        """
        Stack each interval (division) data.

        interval_data_dict: dictionary of interval (division) data. {interval_id: interval_data}
        """
        stacked_data_dict = {}
        for interval_id in interval_data_dict.keys():
            interval_data = np.stack(interval_data_dict[interval_id], 0)
            stacked_data_dict[interval_id] = interval_data.transpose(0,2,1)
        return stacked_data_dict

    def train_compose(self, train_data_list: list):
        """
        Compose the train dataset. The input must be a list of sequential data.

        train_data_list: list of sequential data
        """
        if type(train_data_list) is not list:
            raise Exception("The input dataset must be a list of pd.DataFrame.")

        variable_cols = _find_variable_cols(train_data_list)
        variable_train_data = _remove_constant_cols(train_data_list, variable_cols)
        divided_train_data = self._divide_dataset_list(variable_train_data)
        if self.normalize:
            for divide_id in range(self.divide_num):
                divide_train_data = divided_train_data[divide_id]
                mean, std = _compute_statistics(divide_train_data)
                divided_train_data[divide_id] = _normalize_data(divide_train_data, mean, std)
                statistic_dt = pd.concat([mean, std], axis=1)
                statistic_dt.columns = ["mean", "std"]
                # statistic_dt.to_csv(f"{self.save_dir_name}_divide{divide_id}.csv")
                self.statistic[divide_id] = statistic_dt
        else:
            # pd.DataFrame(index=variable_cols).to_csv(self.save_dir_name)
            self.statistic[0] = pd.DataFrame(index=variable_cols)
        train_input = self._stack_data(divided_train_data)
        return train_input

    def test_compose(self, test_data_dict: OrderedDict):
        """
        Compose the test dataset. The input must be a dictionary of sequential data.

        test_data_dict: dictionary of sequential data
        """
        if type(test_data_dict) is not OrderedDict:
            raise Exception("The input dataset must be a OrderedDict of pd.DataFrame.")
        
        if self.normalize:
            # variable_cols = pd.read_csv(f"{self.save_dir_name}_divide0.csv", index_col=0).index
            variable_cols = self.statistic[0].index
        else:
            # variable_cols = pd.read_csv(f"{self.save_dir_name}.csv", index_col=0).index
            variable_cols = self.statistic[0].index

        variable_train_data = _remove_constant_cols(test_data_dict.values(), variable_cols)
        variable_train_data = OrderedDict({k:variable_train_data[k_id] for k_id, k in enumerate(test_data_dict.keys())})
        divided_test_keys, divided_test_data = self._divide_dataset_dict(variable_train_data)
        if self.normalize:
            for divide_id in range(self.divide_num):
                divide_id_test_data = divided_test_data[divide_id]
                # statistics = pd.read_csv(f"{self.save_dir_name}_divide{divide_id}.csv", index_col=0)
                statistics = self.statistic[divide_id]
                mean, std = statistics["mean"], statistics["std"]
                divided_test_data[divide_id] = _normalize_data(divide_id_test_data, mean, std)
        test_input = self._stack_data(divided_test_data)
        return divided_test_keys, test_input

class DataSlider:
    def __init__(self, input_len: int, output_len: int, normalize: bool):
        """
        Data slider which compose sliding window input-output set.

        input_len: input length
        output_len: output length
        normalize: normalize or not (boolean)
        """
        self.input_len = input_len
        self.output_len = output_len
        self.normalize = normalize
        self.statistic = dict()

    def _slide_data_list(self, data_list: list, stride: int, stack: bool):
        """
        Compose slided input, output, and time, using list of data.

        data_list: list of data
        stride: sliding stride
        stack: numpy stack or not (boolean)
        """
        all_input, all_output, all_times = [], [], []
        for data in data_list:
            data_input, data_output, data_times = _slide_data(data, self.input_len, self.output_len, stride)
            all_input.append(data_input)
            all_output.append(data_output)
            all_times.append(data_times)
        if stack:
            all_input = [dt for dt in all_input if len(dt) > 0]
            all_output = [dt for dt in all_output if len(dt) > 0]
            all_input = np.vstack(all_input)
            all_output = np.vstack(all_output)
        return all_input, all_output, all_times

    def train_compose(self, train_data_list: list, save_dir: str):
        """
        Compose training data.

        train_data_list: list of train data
        save_dir: directory to save mean, std and variable columns
        """
        if type(train_data_list) is not list:
            raise Exception("The input dataset must be a list of pd.DataFrame.")
        variable_cols = _find_variable_cols(train_data_list)
        variable_train_data = _remove_constant_cols(train_data_list, variable_cols)
        if self.normalize:
            mean, std = _compute_statistics(variable_train_data)
            original_train_data = _normalize_data(variable_train_data, mean, std)
            statistic_dt = pd.concat([mean, std], axis=1)
            statistic_dt.columns = ["mean", "std"]
            # statistic_dt.to_csv(save_dir)
            self.statistic = statistic_dt
        else:
            original_train_data = variable_train_data
            # pd.DataFrame(index=variable_cols).to_csv(save_dir)
            self.statistic[0] = pd.DataFrame(index=variable_cols)
        train_input, train_output, _ = self._slide_data_list(original_train_data, 1, True)
        return train_input, train_output
    
    def test_compose(self, test_data_list: list, save_dir: str):
        """
        Compose testing data.

        test_data_list: list of test data
        save_dir: directory of mean, std and variable columns saved
        """
        if type(test_data_list) is not list:
            raise Exception("The input dataset must be a list of pd.DataFrame.")
        # statistics = pd.read_csv(save_dir, index_col=0)
        statistics = self.statistic
        variable_test_data = _remove_constant_cols(test_data_list, list(statistics.index))
        if self.normalize:
            original_test_data = _normalize_data(variable_test_data, statistics["mean"], statistics["std"])
        else:
            original_test_data = variable_test_data
        test_input, test_output, test_times = self._slide_data_list(original_test_data, self.output_len, False)
        return test_input, test_output, test_times

class ErrorAnalyzer:
    def __init__(self, std_multiply: float):
        """
        Train error analyzer.

        std_multiply: multiplier to std for safe range finding
        """
        self.std_multiply = std_multiply

    def compute_error_range(self, train_error: pd.DataFrame):
        """
        Compute the safe range of error. Set the range as 
        (mean - std_multiply*std, mean + std_multiply*std).

        train_error: train data regression error
        """
        error_mean = train_error.mean(axis=0)
        error_std = train_error.std(axis=0)
        upper_range = error_mean + self.std_multiply*error_std
        lower_range = error_mean - self.std_multiply*error_std
        return lower_range, upper_range
    
