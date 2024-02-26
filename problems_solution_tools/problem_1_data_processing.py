"""
This module represents tools and methodologies for solving first problem.

**Problem 1: Python programming, data processing.**
In this problem we want to generate pseudo-random data that has certain desired statistical properties.
This can be useful for demo, research or testing purposes.

First, let’s generate these “desired statistical properties”.
- Generate a random 6x6 correlation matrix rho.
- Regularization: write a test checking that rho is a valid correlation matrix, and if not - find the nearest valid one.
Now, let’s generate the data that would have these properties.
- Generate a set of 6 random variables (put them in a matrix 1000x6, the distribution doesn’t matter,
  but should be continuous), distributed between 0 and 1 with correlation defined by rho.

Slightly different, but related problem.
- Apply PCA to reduce the dimensionality to 5.
- Let the output variable y = round(x6).
- Build a couple of classifiers of your choice to predict y from {x1, x2, …, x5}.
- Compare their performance.
"""

import numpy as np
from numpy import copy
from numpy.linalg import norm
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import logging

from exceptions import ExceededMaxIterationsError, WrongModelStructure


class DataProcessing:
    """
    Class supposed to be used for data generation, normalization, splitting into train / test datasets
    """

    def __int__(self):
        self.correlation_matrix = np.array([])

    def random_correlation_matrix(self, shape: int) -> np.ndarray:
        """
        Generates random correlation matrix with shape {shape}
        :param shape: shape of matrix
        :return: matrix
        """
        random_matrix = np.random.rand(shape, shape)
        self.correlation_matrix, p = spearmanr(
            random_matrix
        )  # Calculating RHO correlation matrix
        if not self.is_valid_correlation_matrix():
            logging.warning(
                "random generated correlation matrix is not valid. Generating nearest valid matrix"
            )
            self.correlation_matrix = (
                self.nearcorr()
            )  # calculating nearest correlation matrix
        return self.correlation_matrix

    def generate_normalized_data_based_correlation_matrix(
        self, count: int = 1000
    ) -> np.ndarray:
        """
        Generates normalized (in range [0, 1]) dataset with shape (count, correlation_matrix.shape[0])
        :param count: size of dataset
        :return: dataset
        """
        nm = MinMaxScaler()
        data = np.random.multivariate_normal(
            mean=np.zeros(self.correlation_matrix.shape[0]),
            cov=self.correlation_matrix,
            size=count,
        )
        data_nm = nm.fit_transform(data)
        return data_nm

    def build_train_test_dataset(self, data: np.ndarray, test_size=0.15) -> tuple:
        """
        Splitting dataset between train and test parts
        :param data: normalized data
        :param test_size: size of testing dataset
        :return: tuple of splitted datasets
        """
        data_x, data_y = data[:, :-1], np.round(data[:, -1])
        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=test_size, stratify=data_y
        )
        return x_train, x_test, y_train, y_test

    def is_valid_correlation_matrix(self, matrix: np.ndarray = None) -> bool:
        """
        Check if the matrix is a valid correlation matrix.
        param: matrix
        return: matrix
        """
        if matrix is None:
            matrix = self.correlation_matrix

        # Check if matrix is square
        if not np.allclose(matrix, matrix.T):
            return False

        # Check if diagonal elements are 1
        if not np.allclose(np.diag(matrix), 1):
            return False

        # Check if matrix is positive definite
        eigenvalues, _ = np.linalg.eigh(matrix)
        if not np.all(eigenvalues >= 0):
            return False
        return True

    def nearcorr(self, max_iterations=100, except_on_too_many_iterations=True):
        """
        Calculates nearest correlation matrix
        :param max_iterations: maximum iterations for finding nearest matrix
        :param except_on_too_many_iterations: flag for interrupting infinite loop
        :return: matrix
        ------
        Example: X = nearcorr(A, max_iterations=100)
        """
        matrix = self.correlation_matrix
        ds = np.zeros(np.shape(matrix))

        eps = np.spacing(1)
        if not np.all((np.transpose(matrix) == matrix)):
            raise ValueError("Input Matrix is not symmetric")
        tol = eps * np.shape(matrix)[0] * np.array([1, 1])
        weights = np.ones(np.shape(matrix)[0])
        X = copy(matrix)
        Y = copy(matrix)
        rel_diffY = np.inf
        rel_diffX = np.inf
        rel_diffXY = np.inf

        Whalf = np.sqrt(np.outer(weights, weights))

        iteration = 0
        while max(rel_diffX, rel_diffY, rel_diffXY) > tol[
            0
        ] or not self.is_valid_correlation_matrix(X):
            iteration += 1
            if iteration > max_iterations:
                if except_on_too_many_iterations:
                    raise ExceededMaxIterationsError(
                        f"No solution (valid nearest correlation matrix) "
                        f"found in {max_iterations} iterations"
                    )
                else:
                    # exceptOnTooManyIterations is false so just silently
                    # return the result even though it has not converged
                    return X

            Xold = copy(X)
            R = X - ds
            R_wtd = Whalf * R
            X = self._proj_spd(R_wtd)
            X = X / Whalf
            ds = X - R
            Yold = copy(Y)
            Y = copy(X)
            np.fill_diagonal(Y, 1)
            normY = norm(Y, "fro")
            rel_diffX = norm(X - Xold, "fro") / norm(X, "fro")
            rel_diffY = norm(Y - Yold, "fro") / normY
            rel_diffXY = norm(Y - X, "fro") / normY

            X = copy(Y)

        return X

    def _proj_spd(self, matrix: np.ndarray) -> np.ndarray:
        d, v = np.linalg.eigh(matrix)
        matrix = (v * np.maximum(d, 0)).dot(v.T)
        matrix = (matrix + matrix.T) / 2
        return matrix


class Classifier:
    def __init__(self, models: list = None):
        if models is None:
            models = [{"name": "LogisticRegression", "estimator": LogisticRegression()}]
        elif type(models) is not list or not all(
            (type(model) == dict for model in models)
        ):
            raise WrongModelStructure(f"provided wrong structure of models.")
        self.models = models
        self.pipelines = []
        self.scores = []

        logging.info(f"Classifier have {len(models)} models")

    def make_classifier_pipelines(self, PCA_components=5) -> None:
        for model in self.models:
            pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_components),
                model["estimator"],
            )
            self.pipelines.append({"pipeline": pipeline, "name": model["name"]})

    def estimate_classifier_pipelines(self, x_train, y_train, cv=5):
        for pipeline in self.pipelines:
            score = cross_val_score(
                estimator=pipeline["pipeline"], X=x_train, y=y_train, cv=cv
            )
            self.scores.append({"name": pipeline["name"], "score": score})
