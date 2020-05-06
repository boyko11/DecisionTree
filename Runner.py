from service.data_service import DataService
from decision_tree_learner import DecisionTreeLearner


class Runner:

    def __init__(self, normalization_method='z'):
        self.decision_tree_learner = DecisionTreeLearner()

    def run(self, k=2):

        feature_data, labels = DataService.get_data("data/play_tennis.csv")
        self.decision_tree_learner.train(feature_data, labels)

        errors = 0
        for test_index in range(labels.size):
            prediction = self.decision_tree_learner.classify(feature_data[test_index, :])
            actual = labels[test_index]

            if prediction != actual:
                errors += 1

        accuracy = (labels.size - errors) / labels.size

        print("Accuracy: ", accuracy)


if __name__ == "__main__":

    Runner().run()
