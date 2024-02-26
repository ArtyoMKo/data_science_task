import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import heatmap, scatterplotmatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class DataPreprocessing:
    def __init__(self):
        self.data = None
        self.score = 0

    def read_file(self, filename: str = "10_01_train_dataset.csv"):
        data = pd.read_csv(filename)
        self.data = data

    def reformat_data_values(self):
        class_re = LabelEncoder()
        non_numeric_columns = list(self.data.select_dtypes(exclude=["number"]).columns)
        for col in non_numeric_columns:
            self.data[col] = class_re.fit_transform(self.data[col].values)

    def plot_correlations(self, cols):
        matrix = np.corrcoef(self.data[cols].values.T)
        heatmap(matrix, figsize=(5, 5), column_names=cols, row_names=cols)
        scatterplotmatrix(
            self.data[cols].values, figsize=(10, 8), names=cols, alpha=0.5
        )
        # plt.tight_layout()
        plt.show()

    def get_data_and_target(self, cols):
        X, y = self.data[cols].values, self.data.iloc[:, -1].values
        return X, y

    def split_train_test_data(self, X, y, test_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def standardize_data(self, X_train, X_test, y_train, y_test):
        st = StandardScaler()
        X_train_std, X_test_std, y_train_std, y_test_std = (
            st.fit_transform(X_train),
            st.fit_transform(X_test),
            st.fit_transform(y_train.reshape(-1, 1)),
            st.fit_transform(y_test.reshape(-1, 1)),
        )
        return X_train_std, X_test_std, y_train_std, y_test_std


class Regressor:
    def __init__(self):
        self.score = None
        self.estimator = DecisionTreeRegressor()

    def train_model(self, X_train_std, y_train_std):
        self.estimator.fit(X_train_std, y_train_std)

    def evaluate_model(self, X_test_std, y_test_std):
        self.score = self.estimator.score(X_test_std, y_test_std)
