import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from problems_solution_tools.problem_1_data_processing import DataProcessing, Classifier
from problems_solution_tools.problem_2_ml_model import DataPreprocessing, Regressor
import logging
from exceptions import ExceededMaxIterationsError, WrongModelStructure


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """
    *********
    Problem 1
    *********
    Description and solving methodo

    1. Generated random correlation matrix based on given shape
    2. Generating normalized data based on correlation matrix generated before
    3. Splitting generated dataset into Train/Test partitions
    4. Choosing multiple ML models for training and testing
    5. Plotting results
    """
    logging.info(f"Starting problem 1 solution")
    data_processing = DataProcessing()
    try:
        corr_matrix = data_processing.random_correlation_matrix(6)
        heatmap(corr_matrix)
        plt.title("Correlation matrix")

        data_nm = data_processing.generate_normalized_data_based_correlation_matrix(
            1000
        )
        x_train, x_test, y_train, y_test = data_processing.build_train_test_dataset(
            data_nm, 0.1
        )

        models = [
            {"name": "LogisticRegression", "estimator": LogisticRegression()},
            {"name": "KNeighborsClassifier", "estimator": KNeighborsClassifier(2)},
            {"name": "MLPClassifier", "estimator": MLPClassifier(alpha=1, max_iter=5)},
        ]
        classifier = Classifier(models)
        classifier.make_classifier_pipelines()
        classifier.estimate_classifier_pipelines(x_train, y_train)

        for score in classifier.scores:
            print(f"Classifier: {score['name']}, Scores: {score['score']}")

        plt.figure(figsize=(10, 6))

        for classifier in classifier.scores:
            plt.plot(classifier["score"], label=classifier["name"])

        plt.title("Classifier Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()
    except ExceededMaxIterationsError as e:
        logging.error(f"Please run again. Error msg: {e}")
    except WrongModelStructure as e:
        logging.error(f"Please run again with valid models structure. Error msg: {e}")

    """
    *********
    Problem 2
    *********
    Description and solving methodo
    
    Based on testing combinations of metrics, dimensional reductions (PCA, LDA ...) etc, I choose multiple columns
    which you can find below. 
    Steps:
    1. Encoding data values from str to numeric (for example region)
    2. Plotting correlations between multiple metrics
    3. Splitting data into X and y (y is target)
    4. Splitting into Train/Test partitions and standardizing them
    5. Training model on Train dataset and evaluating it on Test dataset
    """
    logging.info(f"\nStarting problem 2 solution")

    data_preprocessor = DataPreprocessing()
    data_preprocessor.read_file()
    data_preprocessor.reformat_data_values()

    columns = [
        "target",
        "region",
        "size",
        "pop_count_0_25",
        "org_retail_visits_1_0",
        "day_count_0_75",
        "density_lvl_0_25",
    ]
    data_preprocessor.plot_correlations(columns)

    logging.info(
        f"Couple of metrics have correlations with target metric, which is possible to link"
    )
    X, y = data_preprocessor.get_data_and_target(columns)

    X_train, X_test, y_train, y_test = data_preprocessor.split_train_test_data(X, y)
    X_train_std, X_test_std, y_train_std, y_test_std = (
        data_preprocessor.standardize_data(X_train, X_test, y_train, y_test)
    )

    regressor = Regressor()
    regressor.train_model(X_train_std, y_train_std)
    regressor.evaluate_model(X_test_std, y_test_std)
    logging.info(
        f"Used DecisionTreeRegressor model. "
        f"Trained regressor have {regressor.score}"
    )
