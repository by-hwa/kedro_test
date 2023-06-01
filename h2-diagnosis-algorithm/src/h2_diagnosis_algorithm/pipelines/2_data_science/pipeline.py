from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_encode_model, get_cluster, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # hp node
            node(
                func=train_encode_model,
                inputs=["train_set", "params:data_science_option.asset-hp", "params:data_science_option"],
                outputs=["HP_encode_model", "HP_encode_statistic"],
                name="hp_encode_model"
            ),
            node(
                func=get_cluster,
                inputs="HP_encode_model",
                outputs="HP_cluster",
                name="hp_clutering"
            ),
            node(
                func=train_model,
                inputs=["train_set", "test_set", "HP_cluster", "params:data_science_option.asset-hp", "params:data_science_option"],
                outputs=["HP_model", "HP_statistic",  "HP_pred_value"],
                name="hp_model"
            ),
            # mp node
            node(
                func=train_encode_model,
                inputs=["train_set", "params:data_science_option.asset-mp", "params:data_science_option"],
                outputs=["MP_encode_model", "MP_encode_statistic"],
                name="mp_encode_model"
            ),
            node(
                func=get_cluster,
                inputs="MP_encode_model",
                outputs="MP_cluster",
                name="mp_clutering"
            ),
            node(
                func=train_model,
                inputs=["train_set", "test_set", "MP_cluster", "params:data_science_option.asset-mp", "params:data_science_option"],
                outputs=["MP_model", "MP_statistic", "MP_pred_value"],
                name="mp_model"
            ),
        ]
    )