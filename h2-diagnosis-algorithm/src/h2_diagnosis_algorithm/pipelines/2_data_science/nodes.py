import pandas as pd
from datetime import timedelta, datetime
from typing import Dict, Tuple
from datetime import datetime, timedelta
import sys
import os
# sys.path.append('/Users/leebyeonghwa/workspace/kedro_test/h2-diagnosis-algorithm/src/h2_diagnosis_algorithm')
# sys.path.append('/Users/leebyeonghwa/workspace/kedro_test/h2-diagnosis-algorithm/src/h2_diagnosis_algorithm/modules')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from modules.analyzer.autoencode import EncodeCluster
from modules.asset.hyundai.data import split_train_test_data, DataReader
from modules.asset.hyundai.anomaly_encode import CnnTrainEncode, EncodeAnalyze
from modules.asset.hyundai.future_regression import CnnTrainTest, ErrorJudge


def train_encode_model(train_data, parameters: dict):
    divide_len = parameters['divide-len']
    training = parameters['training']

    hp_train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", "HP", divide_len["HP"], 10, training)
    mp_train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", "MP", divide_len["MP"], 10, training)

    hp_model = hp_train_encode.train_models(train_data, 70, 4)
    mp_model = mp_train_encode.train_models(train_data, 70, 4)

    hp_encode = hp_train_encode.encode_models(train_data, hp_model)
    mp_encode = mp_train_encode.encode_models(train_data, mp_model)

    return hp_encode, mp_encode


def get_cluster(hp_encode, mp_encode):
    encode_cluster = EncodeCluster(10)
    
    hp_cluster = encode_cluster.cluster_encode(hp_encode)
    mp_cluster = encode_cluster.cluster_encode(mp_encode)

    return hp_cluster, mp_cluster


def train_model(train_data: dict, test_data: dict, hp_cluster, mp_cluster, parameters: dict):
    model_name = parameters['model-name']
    divide_len = parameters['divide-len']

    hp_cnn_train_test = CnnTrainTest(model_name, "models/v3_daily/", divide_len, "HP", 15, 5, True, True)
    mp_cnn_train_test = CnnTrainTest(model_name, "models/v3_daily/", divide_len, "MP", 15, 5, True, True)

    hp_model = hp_cnn_train_test.divdie_train_asset(train_data, hp_cluster, 20, 32)
    mp_model = mp_cnn_train_test.divdie_train_asset(train_data, mp_cluster, 20, 32)

    hp_pred_train, hp_true_train, _ = hp_cnn_train_test.divdie_test_models(train_data)
    hp_pred_test, hp_true_test, hp_time_test = hp_cnn_train_test.divdie_test_models(test_data)
    mp_pred_train, mp_true_train, _ = mp_cnn_train_test.divdie_test_models(train_data)
    mp_pred_test, mp_true_test, mp_time_test = mp_cnn_train_test.divdie_test_models(test_data)

    return hp_model, mp_model


# def asset_train_test(train_data: dict, test_data: dict, clusters: dict, parameters: dict):
#     model_name = parameters['model-name']
#     divide_len = parameters['divide-len']

#     plot_save_dir = f"figures/result/{model_name}_divide/v3_daily/"
    
#     error_judge = ErrorJudge(true_train, pred_train, asset, f"models/v3_daily/{model_name}_divide/", 0.5, 60)
#     test_error = error_judge.compute_test_error(true_test, pred_test)
#     test_outlie_rate, test_judge = error_judge.compute_test_results(test_error, time_test)
#     error_judge.plot_data_error(true_test, pred_test, test_error, time_test, plot_save_dir)
#     error_judge.plot_results(test_outlie_rate, test_judge, plot_save_dir)

#     return models['HP'], models['MP']
