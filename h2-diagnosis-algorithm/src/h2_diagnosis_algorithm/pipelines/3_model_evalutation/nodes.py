import pandas as pd
from datetime import timedelta, datetime
from typing import Dict, Tuple
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.analyzer.autoencode import EncodeCluster
from modules.asset.hyundai.data import split_train_test_data, DataReader
from modules.asset.hyundai.anomaly_encode import CnnTrainEncode, EncodeAnalyze
from modules.asset.hyundai.future_regression import CnnTrainTest, ErrorJudge


def error_judge(statistic:pd.DataFrame, pred_value: dict, asset: str, parameters: dict):
    model_name = parameters['model-name']

    pred_train = pred_value["pred_train"]
    true_train = pred_value["true_train"]
    pred_test = pred_value["pred_test"]
    true_test = pred_value["true_test"]
    time_test = pred_value["time_test"]

    plot_save_dir = f"figures/result/{model_name}_divide/v3_daily/"
    
    error_judge = ErrorJudge(true_train, pred_train, asset, f"models/v3_daily/{model_name}_divide/", 0.5, 60, statistic)
    test_error = error_judge.compute_test_error(true_test, pred_test)
    test_outlie_rate, test_judge = error_judge.compute_test_results(test_error, time_test)

    return error_judge.plot_data_error(true_test, pred_test, test_error, time_test, plot_save_dir), error_judge.plot_results(test_outlie_rate, test_judge, plot_save_dir)
