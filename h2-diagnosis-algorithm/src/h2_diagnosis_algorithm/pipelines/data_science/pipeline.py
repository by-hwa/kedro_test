from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_encode_model, get_cluster, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_encode_model,
                inputs=["train_set", "params:data_science_option"],
                outputs=["HP_trained_model", "MP_trained_model", "HP_encode_model", "MP_encode_model"],
                name="train_encode_model"
            ),
            node(
                func=get_cluster,
                inputs=["HP_encode_model", "MP_encode_model"],
                outputs=["HP_cluster", "MP_cluster"],
                name="clutering"
            ),
            node(
                func=train_model,
                inputs=["train_set", "test_set", "HP_cluster", "MP_cluster", "params:data_science_option"],
                outputs=["HP_model", "MP_model"],
                name="train_model"
            ),
        ]
    )