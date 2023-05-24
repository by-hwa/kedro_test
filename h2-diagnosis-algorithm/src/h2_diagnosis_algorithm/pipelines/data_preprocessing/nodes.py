import pandas as pd
from datetime import timedelta, datetime
from typing import Dict, Tuple
import os

## data split
def train_test_names(parameters: Dict):
    train_start, train_end, test_start, test_end = parameters['split-time']
    date_name = train_start
    train_names = []
    while date_name <= train_end:
        date_str = date_name.strftime("%Y-%m-%d")
        train_names.append(date_str)
        date_name += timedelta(days=1)
    test_names = []

    date_name = test_start
    while date_name <= test_end:
        date_str = date_name.strftime("%Y-%m-%d")
        test_names.append(date_str)
        date_name += timedelta(days=1)
    data_split = {"train":train_names, "test":test_names}
    return data_split


## data_reader
def _find_asset_cols(all_cols, find_cols: list):
    if len(find_cols) < 1:
        return list(all_cols)
    plot_cols = []
    for col_n in find_cols:
        plot_cols.extend([c for c in all_cols if col_n in c.lower()])
    return plot_cols


def _remove_parts(all_cols: list, exclude_parts: list):
    """
    Remove the part of columns whose name contains excluding parts from the original columns.

    all_cols: all original columns
    exclude_parts: parts to remove from columns
    """
    if len(exclude_parts) < 1:
        return all_cols
    remove_cols = _find_asset_cols(all_cols, exclude_parts)
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


def _find_assets(all_dataset: Dict, asset_types: list):
        """
        Find assets of asset type.

        asset_types: list of asset types
        """
        all_assets = []
        for dt in all_dataset.values():
            dt_assets = _extract_asset_name(dt.columns)
            new_assets = [a for a in dt_assets if a not in all_assets]
            all_assets.extend(new_assets)
        type_assets = []
        for asset in asset_types:
            correspond_asset = [a for a in all_assets if asset in a]
            type_assets.extend(correspond_asset)
        return type_assets


def find_running_part(parameters: Dict):
        """
        Find running part, where the running variable is continuously 1 ( >0 ).
        """
        all_dataset = _read_data_path(data_path=parameters['file-path'], exclude_parts=parameters['exclude-part'])
        assets = _find_assets(all_dataset=all_dataset, asset_types=parameters['asset-type'])

        all_running_dt = {}
        for dt_name in all_dataset.keys():
            all_running_dt[dt_name] = {}
            dt = all_dataset[dt_name]
            for asset in assets:
                asset_cols = [c for c in dt.columns if asset in c]
                asset_running_list = [c for c in asset_cols if 'running' in c.lower()]
                if len(asset_running_list) > 0: # If running=0 in the whole, already removed in "_read_data_path".
                    running_asset_dt = dt[asset_cols].loc[dt[asset_running_list[0]]>0]
                    all_running_dt[dt_name][asset] = running_asset_dt
        return all_running_dt

## train_data, test_data
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
