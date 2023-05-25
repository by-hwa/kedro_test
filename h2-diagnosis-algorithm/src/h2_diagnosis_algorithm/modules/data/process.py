from datetime import timedelta

def find_intervals(dt_index):
    end_points = list(dt_index[:-1][dt_index[1:] - dt_index[:-1] > timedelta(seconds=2)]) + [dt_index[-1]]
    start_points = [dt_index[0]] + list(dt_index[1:][dt_index[1:] - dt_index[:-1] > timedelta(seconds=2)])
    all_intervals = []
    for i in range(len(start_points)):
        all_intervals.append((start_points[i], end_points[i]))
    return all_intervals

def find_asset_cols(all_cols, find_cols: list):
    if len(find_cols) < 1:
        return list(all_cols)
    plot_cols = []
    for col_n in find_cols:
        plot_cols.extend([c for c in all_cols if col_n in c.lower()])
    return plot_cols
