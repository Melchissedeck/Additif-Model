# data_preparation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreparation:
    def __init__(self, csv_path):
        self.dataset_df = pd.read_csv(csv_path)
        self.dataset_df["Years"] = pd.to_datetime(self.dataset_df["Years"])
        self.prepare_data()

    def prepare_data(self):
        self.dataset_df['month_name'] = self.dataset_df['Years'].dt.month
        self.dataset_df = pd.get_dummies(self.dataset_df, columns=['month_name'],prefix='month')

        number_of_rows = len(self.dataset_df)
        self.dataset_df["index_mesure"] = np.arange(0, number_of_rows, 1)

        dataset_train_df = self.dataset_df.iloc[:int(number_of_rows * 0.75)]
        dataset_test_df = self.dataset_df.iloc[int(number_of_rows * 0.75):]



        self.x_train = dataset_train_df[['index_mesure', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']].values
        self.y_train = dataset_train_df[['Sales']].values

        self.x_test = dataset_test_df[['index_mesure', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']].values
        self.y_test = dataset_test_df[['Sales']].values

    def show_graph(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.dataset_df["Years"], self.dataset_df["Sales"], "o:")
        plt.show()


    def display_dataframe(self):
        print(self.dataset_df)

