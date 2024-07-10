import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataProcessing:
    def __init__(self, df):
        self.df = df

    def identify_outliers(self, data: pd.DataFrame):
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_title("box plot of data")
        ax.set_ylabel("value")
        plt.show()

    def identify_outliers_zscore(self, data: pd.Series, threshold: float = 3):
        mean = np.mean(data)
        std = np.std(data)
        z_score = (data - mean)/std
        outliers = data[np.abs(z_score) > threshold]
        return outliers
