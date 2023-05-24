from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test1

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
            func=test1,
            inputs="raw_datas",
            outputs=None,
            name="test_node1"
            )
        ]
    )