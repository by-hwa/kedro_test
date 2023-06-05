from kedro.pipeline import Pipeline, node, pipeline

from .nodes import error_judge

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=error_judge,
                inputs=["HP_statistic", "HP_pred_value", "params:evaluation_option.asset-hp", "params:evaluation_option"],
                outputs=["hp_plot_data_error", "hp_plot_result"],
                name="hp_error_judge"
                ),
            node(
            func=error_judge,
            inputs=["MP_statistic", "MP_pred_value", "params:evaluation_option.asset-mp", "params:evaluation_option"],
            outputs=["mp_plot_data_error", "mp_plot_result"],
            name="mp_error_judge"
            )
        ]
    )