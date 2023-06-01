import pandas as pd
from datetime import timedelta, datetime
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from modules.asset.hyundai.data import split_train_test_data, DataReader

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
def find_running_part(parameters: Dict):
    """
    Find running part, where the running variable is continuously 1 ( >0 ).
    """

    data_reader = DataReader(parameters['file-path'], parameters['asset-type'], parameters['exclude-part'])
    running_dt = data_reader.find_running_part()

    return running_dt

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
