from kedro.pipeline import Pipeline, node, pipeline

from .nodes import asset_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
            func=asset_train_test,
            inputs=["train_set", "test_set", "params:data_science_option"],
            outputs=["HP_model", "MP_model"],
            name="train_model"
            ),
        ]
    )