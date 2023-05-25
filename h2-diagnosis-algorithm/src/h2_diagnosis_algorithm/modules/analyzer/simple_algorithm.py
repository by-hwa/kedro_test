import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

class SimpleAlgorithm:
    def __init__(self, std_window: int):
        self.std_window = std_window
        
    def train(self, data_list: list, save_dir: str):
        sliding_std = {}
        for data in data_list:
            for col in data.columns:
                if col not in sliding_std.keys():
                    sliding_std[col] = []
                col_sliding_view = sliding_window_view(data[col], self.std_window)
                stride_sliding_view = col_sliding_view[range(0,len(col_sliding_view), self.std_window)]
                col_sliding_fluctuation = stride_sliding_view.max(axis=1) - stride_sliding_view.min(axis=1)
                sliding_std[col].append(col_sliding_fluctuation)
        std_statistics = {}
        for col in sliding_std.keys():
            col_sliding_std = np.concatenate(sliding_std[col])
            if col_sliding_std.max() - col_sliding_std.mean() == 0:
                continue
            std_statistics[col] = {"mean": np.mean(col_sliding_std), "std": np.std(col_sliding_std)}
        std_statistics_df = pd.DataFrame(std_statistics).T
        std_statistics_df.to_csv(f"{save_dir}.csv")

    def test(self, data_list: list, save_dir: str):
        std_statistics = pd.read_csv(save_dir, index_col=0)
        all_judges = []
        for data in data_list:
            data_judge = pd.DataFrame(
                columns=std_statistics.index,
                index=data.index[range(self.std_window-1, len(data), self.std_window)]
            )
            for col in std_statistics.index:
                col_sliding_view = sliding_window_view(data[col], self.std_window)
                stride_sliding_view = col_sliding_view[range(0, len(col_sliding_view), self.std_window)]
                col_sliding_fluctuation = stride_sliding_view.max(axis=1) - stride_sliding_view.min(axis=1)
                col_mean, col_std = std_statistics.loc[col, "mean"], std_statistics.loc[col, "std"]
                data_judge[col] = (np.abs(col_sliding_fluctuation-col_mean) > 3*col_std).astype(int)
            all_judges.append(data_judge)
        return all_judges
