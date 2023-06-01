from datetime import datetime, timedelta
import calendar, os
import numpy as np
import matplotlib.pyplot as plt
from modules.asset.hyundai.data import split_interval_list, split_interval_dict

class DatePlot:
    def __init__(self, raw_data, outlie_rate, judges, dates: list, cut_len: int, save_dir: str):
        self.raw_data = raw_data
        self.outlie_rate = outlie_rate
        self.judges = judges
        self.dates = dates
        self.cut_len = cut_len
        self.save_dir = save_dir
        self.anomaly_dates = self._find_anomaly_dates()

    def _find_anomaly_dates(self):
        anomaly_dates = []
        for asset in self.judges.keys():
            asset_judge = self.judges[asset]
            for date in asset_judge.keys():
                if sum(asset_judge[date].values()) > 0:
                    convert_date = datetime.strptime(date, '%Y-%m-%d')
                    anomaly_dates.append(convert_date)
        anomaly_dates = np.unique(anomaly_dates)
        return anomaly_dates

    def plot_calendar(self, save_path: str):
        month_range = range(self.dates[0].month, self.dates[-1].month + 1)
        year = 2023
        for month in month_range:
            month_anomaly_days = [d.day for d in self.anomaly_dates if d.month==month]
            cal_matrix = calendar.monthcalendar(year, month)

            fig, ax = plt.subplots()
            ax.set_xlim(0, 7)
            ax.set_ylim(0, 6)
            ax.set_xticks([])
            ax.set_yticks([])
            for week_num, week in enumerate(cal_matrix):
                for day_num, day in enumerate(week):
                    if day < 1:
                        continue
                    if day in month_anomaly_days:
                        ax.text(day_num + 0.5, 6 - week_num - 0.5, str(day), ha='center', va='center', color='red')
                    else:
                        ax.text(day_num + 0.5, 6 - week_num - 0.5, str(day), ha='center', va='center')

            month_name = calendar.month_name[month]
            ax.set_title(f"{month_name} {year}")
            plt.savefig(f"{save_path}-{month_name}")
            plt.close()

    def plot_anomaly_data(self, save_path: str):
        interval_dataset = split_interval_dict(self.raw_data, self.cut_len)
        for asset in self.judges.keys():
            asset_judge = self.judges[asset]
            for date in asset_judge.keys():
                date_judge = asset_judge[date]
                for interval_id in date_judge.keys():
                    interval_abnormal = date_judge[interval_id]
                    if not interval_abnormal:
                        continue
                    interval_dt = interval_dataset[asset][date][interval_id]
                    interval_outlie = self.outlie_rate[asset][date][interval_id]
                    fig, axs = plt.subplots(2,1,figsize=(8,12))
                    axs[0].plot(interval_dt)
                    axs[0].axvline(interval_dt.index[0]+timedelta(seconds=self.cut_len), color='r', linestyle='--')
                    axs[0].legend(interval_dt.columns)
                    axs[1].plot(interval_outlie)
                    axs[1].set_ylim((-0.1, 1.1))
                    fig_path = f"{save_path}/{date}/"
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    plt.savefig(f"{fig_path}{asset} interval{interval_id}")
                    plt.close()
