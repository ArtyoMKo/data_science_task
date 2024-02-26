import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from data_processing_1_exc import DataProcessing, Classifier
import logging
from exceptions import ExceededMaxIterationsError, WrongModelStructure

if __name__ == "__main__":
    data_processing = DataProcessing()
    try:
        corr_matrix = data_processing.random_correlation_matrix(6)
        heatmap(
            corr_matrix
        )
        plt.title('Correlation matrix')

        data_nm = data_processing.generate_normalized_data_based_correlation_matrix(1000)
        x_train, x_test, y_train, y_test = data_processing.build_train_test_dataset(data_nm, 0.1)

        models = [
            {
                'name': 'LogisticRegression',
                'estimator': LogisticRegression()
            },
            {
                'name': 'KNeighborsClassifier',
                'estimator': KNeighborsClassifier(2)
            },
            {
                'name': 'MLPClassifier',
                'estimator':  MLPClassifier(alpha=1, max_iter=5)
            }
        ]
        classifier = Classifier(models)
        classifier.make_classifier_pipelines()
        classifier.estimate_classifier_pipelines(x_train, y_train)

        for score in classifier.scores:
            print(f"Classifier: {score['name']}, Scores: {score['score']}")

        plt.figure(figsize=(10, 6))

        for classifier in classifier.scores:
            plt.plot(classifier['score'], label=classifier['name'])

        plt.title('Classifier Scores')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ExceededMaxIterationsError as e:
        logging.error(f"Please run again. Error msg: {e}")
    except WrongModelStructure as e:
        logging.error(f"Please run again with valid models structure. Error msg: {e}")
