from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_test_names, find_running_part, split_train_test_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_names,
                inputs="params:preprocessing_option",
                outputs="data_split",
                name="train_test_names",
            ),
            node(
                func=find_running_part,
                inputs="params:preprocessing_option",
                outputs="running_dt",
                name="find_running_part",
            ),
            node(
                func=split_train_test_data,
                inputs=["running_dt", "data_split"],
                outputs=["train_set", "test_set"],
            ),
        ]
    )