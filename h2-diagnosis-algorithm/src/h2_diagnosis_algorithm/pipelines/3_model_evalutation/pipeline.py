from kedro.pipeline import Pipeline, node, pipeline

from .nodes import error_judge

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=error_judge,
                inputs=["HP_statistic", "HP_pred_value", "params:evaluation_option.asset-hp", "params:evaluation_option"],
                outputs=None,
                name="hp_error_judge"
                )
        ]
    )