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

from modules.data.compressor import DataDivider
from modules.analyzer.autoencode import DivideEncoder
from modules.asset.hyundai.data import split_interval_list, split_interval_dict, remove_asset_part

def train_encode_model(train_data, asset: str, parameters: dict):
    divide_len = parameters['divide-len']
    training = parameters['training']

    train_encode = CnnTrainEncode("models/Autoencoder/regression_part/", asset, divide_len[asset], 10, training)

    model, statistic = train_encode.train_models(train_data, 70, 4)

    encode = train_encode.encode_models(train_data, model)

    return encode, statistic


def get_cluster(encode):
    encode_cluster = EncodeCluster(10)
    
    cluster = encode_cluster.cluster_encode(encode)

    return cluster


def train_model(train_data: dict, test_data: dict, cluster, asset: str, parameters: dict):
    model_name = parameters['model-name']
    divide_len = parameters['divide-len']

    cnn_train_test = CnnTrainTest(model_name, "models/v3_daily/", divide_len, asset, 15, 5, True, True)

    # model = cnn_train_test.divdie_train_asset(train_data, cluster, 20, 32)

    # test
    model, statistic = cnn_train_test.divdie_train_asset(train_data, cluster, 1, 32)

    pred_train, true_train, _ = cnn_train_test.divdie_test_models(train_data)
    pred_test, true_test, time_test = cnn_train_test.divdie_test_models(test_data)

    return model, statistic, {"pred_train":pred_train, "true_train":true_train, "pred_test":pred_test, "true_test":true_test, "time_test":time_test}
