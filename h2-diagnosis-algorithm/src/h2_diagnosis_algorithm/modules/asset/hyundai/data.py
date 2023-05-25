import os
from datetime import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from data.process import find_intervals, find_asset_cols

def _find_interval_set(dataset: pd.DataFrame, min_len: int):
    """
    Find the interval dataset where the index is continuous. If the length of interval is less or equal to min_len, remove.

    dataset: interval subtraction object dataset
    min_len: minimum length of interval
    """
    interval_dataset = {}
    all_intervals = find_intervals(dataset.index)
    for interval_id, interval in enumerate(all_intervals):
        interval_dt = dataset[interval[0]:interval[1]]
        if len(interval_dt) > min_len:
            interval_dataset[interval_id] = interval_dt
    return interval_dataset

def _remove_parts(all_cols: list, exclude_parts: list):
    """
    Remove the part of columns whose name contains excluding parts from the original columns.

    all_cols: all original columns
    exclude_parts: parts to remove from columns
    """
    if len(exclude_parts) < 1:
        return all_cols
    remove_cols = find_asset_cols(all_cols, exclude_parts)
    rest_cols = [c for c in all_cols if c not in remove_cols]
    return rest_cols

def _read_data_path(data_path: str, exclude_parts: list):
    """
    Read dataset from the path. Remove the constant columns, including running column.

    data_path: path of data to read
    exclude_parts: parts to remove from columns
    """
    data_files = os.listdir(data_path)
    all_dt = {}
    for file in data_files:
        file_type = file.split('.')[-1]
        if file_type == "csv":
            file_dt = pd.read_csv(data_path+file, index_col=0)
            file_dt["timestamp"] = pd.to_datetime(file_dt["timestamp"])
            file_dt.set_index("timestamp", inplace=True)
        elif file_type == "parquet":
            file_dt = pd.read_parquet(data_path+file)
        variable_cols = file_dt.columns[file_dt.std()>0]
        variable_cols = _remove_parts(variable_cols, exclude_parts)
        all_dt[file.split('.')[0]] = file_dt[variable_cols]
    return all_dt

def _extract_asset_name(columns):
    """
    Extract the asset names, removing the asset types such as [HPC-301A].

    columns: all columns
    """
    all_assets = []
    for col in columns:
        if '[' not in col:
            continue
        asset_name = col[col.index('[')+1 : col.index(']')]
        if asset_name not in all_assets:
            all_assets.append(asset_name)
    return all_assets

def split_interval_list(data_dict: dict, min_len: int):
    """
    Split the sequence data and return as list of interval data.

    data_dict: dictionary of sequence data
    min_len: minimum length of interval
    """
    all_asset_data = OrderedDict()
    for data_name in data_dict.keys():
        asset_data = data_dict[data_name]
        for asset in asset_data.keys():
            if asset not in all_asset_data.keys():
                all_asset_data[asset] = []
            interval_dataset = _find_interval_set(asset_data[asset], min_len)
            all_asset_data[asset].extend(list(interval_dataset.values()))
    return all_asset_data

def split_interval_dict(data_dict: dict, min_len: int):
    """
    Split the sequence data and return as dictrionary of interval data.

    data_dict: dictionary of sequence data
    min_len: minimum length of interval
    """
    all_asset_data = OrderedDict()
    for data_name in data_dict.keys():
        asset_data = data_dict[data_name]
        for asset in asset_data.keys():
            if asset not in all_asset_data.keys():
                all_asset_data[asset] = OrderedDict()
            interval_dataset = _find_interval_set(asset_data[asset], min_len)
            all_asset_data[asset][data_name] = interval_dataset
    return all_asset_data

def list_to_dict(list_data: list, key_list: dict):
    """
    Convert the list of data to dictionary.

    list_data: list of data
    key_list: list of dictionary keys
    """
    dict_data = OrderedDict()
    element_id = 0
    for data_name in key_list.keys():
        dict_data[data_name] = OrderedDict()
        for interval_id in key_list[data_name]:
            dict_data[data_name][interval_id] = list_data[element_id]
            element_id += 1
    return dict_data

def read_variable_cols(save_dir):
    """
    Read the variable columns of each asset.

    save_dir: directory where the columns are saved
    """
    all_col_files = [f for f in os.listdir(save_dir) if f.split('.')[-1]=="csv"]
    asset_variable_cols = OrderedDict()
    for file in all_col_files:
        asset_name = file.split('.')[0]
        asset_variable_cols[asset_name] = list(pd.read_csv(save_dir+file, index_col=0).index)
    return asset_variable_cols

def reform_window_data(window_data: np.ndarray, column_names: list, index=None):
    """
    Reform the sliding window data to flat sequence data. (3D to 2D)

    window_data: data to reformed
    column_names: name of columns
    index: index of sequence
    """
    reform_window_data = window_data.transpose((0, 2, 1)).reshape((window_data.shape[0]*window_data.shape[2], window_data.shape[1]))
    if index is None:
        index = range(len(reform_window_data))
    reform_window_data = pd.DataFrame(reform_window_data, columns=column_names, index=index)
    return reform_window_data

def split_train_test_data(data_dict: dict, split_name: dict):
    """
    Split the train and test dataset.

    data_dict: dictionary of data
    split_name: dictionary of train-test keys
    """
    train_set, test_set = {}, {}
    for dt_name in split_name["train"]:
        train_set[dt_name] = data_dict[dt_name]
    for dt_name in split_name["test"]:
        test_set[dt_name] = data_dict[dt_name]
    return train_set, test_set

def remove_asset_part(df_list: list):
    """
    Remove the asset names (e.g. HPC-301B, MPC-301A) of the columns.

    df_list: list of data
    """
    removed_list = []
    for df in df_list:
        df_copy = df.copy()
        removed_col = [c.split('] ')[-1] for c in df.columns]
        df_copy.columns = removed_col
        removed_list.append(df_copy)
    return removed_list

def _read_column(column_path):
    """
    Read the columns.

    column_path: path where columns are saved
    """
    column_names = pd.read_csv(column_path).dropna()
    column_names['fieldname'] = ['plc@'+c for c in column_names['fieldname']]
    return column_names

def _read_hour_data(date_data_path: str):
    """
    Read the hourly data of a date.

    date_data_path: path to read data
    """
    all_dt = []
    all_hours = [f for f in os.listdir(date_data_path) if f[0] != '.']
    for hour in all_hours:
        hour_dir = os.path.join(date_data_path, hour)
        file_lists = [x for x in os.listdir(hour_dir) if x.split('.')[-1] == 'parquet' and x[0] != '.']
        for file in file_lists:
            file_dir = os.path.join(hour_dir, file)
            file_dt = pd.read_parquet(file_dir).set_index("datetime")
            all_dt.append(file_dt)
    all_dt = pd.concat(all_dt).sort_index()
    return all_dt

def _remove_repetitive_cols(fields_cols):
    """
    Remove the repetitive columns.

    fields_cols: field columns
    """
    cols, col_nums = np.unique(fields_cols["colname"], return_counts=True)
    unique_cols = cols[col_nums < 2]
    unique_fields_cols = fields_cols.loc[np.isin(fields_cols["colname"], unique_cols)]
    return unique_fields_cols

class DataReader:
    def __init__(self, data_path: str, asset_types: list, exclude_parts=[]):
        """
        Read the data and extract running intervals of each asset.

        data_path: path to read csv or parquet files.
        asset_types: Types of asset to extract the columns. For example, if ["HP", "MP"], assets are ['MPC-301B', 'MPC-301A', 'MPC-301C', 'HPC-301B', 'HPC-301C', 'HPC-301A'].
        exclude_parts: part of column names to exclude
        """
        self.all_dataset = _read_data_path(data_path, exclude_parts)
        self.assets = self._find_assets(asset_types)

    def _find_assets(self, asset_types: list):
        """
        Find assets of asset type.

        asset_types: list of asset types
        """
        all_assets = []
        for dt in self.all_dataset.values():
            dt_assets = _extract_asset_name(dt.columns)
            new_assets = [a for a in dt_assets if a not in all_assets]
            all_assets.extend(new_assets)
        type_assets = []
        for asset in asset_types:
            correspond_asset = [a for a in all_assets if asset in a]
            type_assets.extend(correspond_asset)
        return type_assets
    
    def find_running_part(self):
        """
        Find running part, where the running variable is continuously 1 ( >0 ).
        """
        all_running_dt = {}
        for dt_name in self.all_dataset.keys():
            all_running_dt[dt_name] = {}
            dt = self.all_dataset[dt_name]
            for asset in self.assets:
                asset_cols = [c for c in dt.columns if asset in c]
                asset_running_list = [c for c in asset_cols if 'running' in c.lower()]
                if len(asset_running_list) > 0: # If running=0 in the whole, already removed in "_read_data_path".
                    running_asset_dt = dt[asset_cols].loc[dt[asset_running_list[0]]>0]
                    all_running_dt[dt_name][asset] = running_asset_dt
        return all_running_dt

class HourDataConvertor:
    def __init__(self, data_path: str, column_path: str):
        """
        Convert houly data to daily data.

        data_path: path where hourly data is saved
        column_path: path where column names are saved
        """
        self.data_path = data_path
        self.columns = _remove_repetitive_cols(_read_column(column_path))
        
    def convert_data(self, save_dir: str):
        """
        Convert the hourly data to daily data and save.

        save_dir: directory to save converted daily data
        """
        path_dates = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        for date in path_dates:
            date_info = date.split('=')[-1]
            date_data = _read_hour_data(os.path.join(self.data_path, date))
            exist_cols = [c for c in date_data.columns if c in list(self.columns["fieldname"])]
            exist_col_names = self.columns.set_index("fieldname").loc[exist_cols]
            date_data = date_data[exist_cols]
            date_data.columns = exist_col_names["colname"]
            date_data.to_parquet(f"{save_dir}{date_info}.parquet")

class MinuteDataConvertor:
    def __init__(self, data_path: str, column_path: str):
        """
        Convert minute data to daily data.

        data_path: path where minute data is saved
        column_path: path where column names are saved
        """
        self.data_path = data_path
        self.columns = _remove_repetitive_cols(_read_column(column_path))
        
    def convert_data(self, save_dir: str):
        """
        Convert the minute data to daily data and save.

        save_dir: directory to save converted daily data
        """
        year = 2023
        path_months = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        for month_dir in path_months:
            month = int(month_dir.split('=')[-1])
            month_full_dir = os.path.join(self.data_path, month_dir)
            path_days = os.listdir(month_full_dir)
            for day_dir in path_days:
                day = int(day_dir.split('=')[-1])
                date_info = datetime(year, month, day).strftime("%Y-%m-%d")
                day_full_dir = os.path.join(month_full_dir, day_dir)
                path_hours = os.listdir(day_full_dir)
                date_data = []
                for hour_dir in path_hours:
                    hour_full_dir = os.path.join(day_full_dir, hour_dir)
                    path_minutes = os.listdir(hour_full_dir)
                    for minute_dir in path_minutes:
                        minute_full_dir = os.path.join(hour_full_dir, minute_dir)
                        file_lists = [x for x in os.listdir(minute_full_dir) if x.split('.')[-1] == 'parquet' and x[0] != '.']
                        for file in file_lists:
                            file_dir = os.path.join(minute_full_dir, file)
                            file_dt = pd.read_parquet(file_dir).set_index("datetime")
                            date_data.append(file_dt)
                date_data = pd.concat(date_data).sort_index()
                exist_cols = [c for c in date_data.columns if c in list(self.columns["fieldname"])]
                exist_col_names = self.columns.set_index("fieldname").loc[exist_cols]
                date_data = date_data[exist_cols]
                date_data.columns = exist_col_names["colname"]
                date_data.to_parquet(f"{save_dir}{date_info}.parquet")
