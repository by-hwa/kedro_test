import pandas as pd
from datetime import timedelta, datetime
from typing import Dict, Tuple
from datetime import datetime, timedelta
import sys
# sys.path.append('/Users/leebyeonghwa/workspace/kedro_test/h2-diagnosis-algorithm/src/h2_diagnosis_algorithm')
sys.path.append('/Users/leebyeonghwa/workspace/kedro_test/h2-diagnosis-algorithm/src/h2_diagnosis_algorithm/modules')

from analyzer.autoencode import EncodeCluster
from asset.hyundai.data import split_train_test_data, DataReader
from asset.hyundai.anomaly_encode import CnnTrainEncode, EncodeAnalyze
from asset.hyundai.future_regression import CnnTrainTest, ErrorJudge


def train_encode_model(train_data, parameters: dict):
    divide_len = parameters['divide-len']
    training = parameters['training']

    hp_train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", "HP", divide_len["HP"], 10, training)
    mp_train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", "MP", divide_len["MP"], 10, training)

    hp_model = hp_train_encode.train_models(train_data, 70, 4)
    mp_model = mp_train_encode.train_models(train_data, 70, 4)

    hp_encode = hp_train_encode.encode_models(train_data, hp_model)
    mp_encode = mp_train_encode.encode_models(train_data, mp_model)

    return hp_model, mp_model, hp_encode, mp_encode


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

    return hp_model, mp_model


# def cluster_sequences(train_data: dict, asset_type: str, divide_len: int, training: bool):
#     plot_save_dir = "figures/result/Encode/regression_part/"
#     train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", asset_type, divide_len, 10, True)
#     if training:
#         train_encode.train_models(train_data, 70, 4)
#     encode = train_encode.encode_models(train_data)
#     encode_cluster = EncodeCluster(10)
#     cluster = encode_cluster.cluster_encode(encode)
#     if training:
#         analysis = EncodeAnalyze(train_data, divide_len, 10, plot_save_dir)
#         analysis.cluster_data_result(cluster)
#     return cluster




# def asset_train_test(train_data: dict, test_data: dict, clusters: dict, parameters: dict):
    # model_name = parameters['model-name']
    # divide_len = parameters['divide-len']
    # training = parameters['training']
    # # cluster_training = parameters['cluster-training']

    # asset_types = ['HP', 'MP']
    # models = dict()

    # for asset in asset_types:
    #     plot_save_dir = f"figures/result/{model_name}_divide/v3_daily/"
    #     cnn_train_test = CnnTrainTest(model_name, "models/v3_daily/", divide_len, asset, 15, 5, True, True)
    #     if training:
    #         # clusters[asset] = _cluster_sequences(train_data, asset, divide_len[asset], cluster_training)
    #         models[asset] = cnn_train_test.divdie_train_asset(train_data, clusters[asset], 20, 32)
    #     pred_train, true_train, _ = cnn_train_test.divdie_test_models(train_data)
    #     pred_test, true_test, time_test = cnn_train_test.divdie_test_models(test_data)
    
    # # error_judge = ErrorJudge(true_train, pred_train, asset, f"models/v3_daily/{model_name}_divide/", 0.5, 60)
    # # test_error = error_judge.compute_test_error(true_test, pred_test)
    # # test_outlie_rate, test_judge = error_judge.compute_test_results(test_error, time_test)
    # # error_judge.plot_data_error(true_test, pred_test, test_error, time_test, plot_save_dir)
    # # error_judge.plot_results(test_outlie_rate, test_judge, plot_save_dir)

    # return models['HP'], models['MP']
