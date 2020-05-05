from service.data_service import DataService
from decision_tree_learner import DecisionTreeLearner


class Runner:

    def __init__(self, normalization_method='z'):
        self.decision_tree_learner = DecisionTreeLearner()

    def run(self, k=2):

        feature_data, labels = DataService.get_data("data/car.data")
        self.decision_tree_learner.train(feature_data, labels)

        print("--")


if __name__ == "__main__":

    Runner().run()
