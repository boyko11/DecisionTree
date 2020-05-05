import numpy as np


class DecisionTreeLearner:

    def __init__(self):
        self.decision_tree = None
        self.label_probabilities = {}
        self.initial_label_entropy = None
        self.initial_label_entropy_with_respect_to_features = None

    def train(self, feature_data, labels):

        # calculate probability of labels
        unique_labels, counts_per_label = np.unique(labels, return_counts=True)
        self.initial_label_entropy = self.entropy(counts_per_label)

        # for each feature
        attr_entropies = []
        for attr_index in range(feature_data.shape[1]):
            # get this attribute's unique values
            unique_attribute_values, counts_per_value = np.unique(feature_data[:, attr_index], return_counts=True)
            attr_val_probabilities = counts_per_value / np.sum(counts_per_value)
            label_entropies_with_respect_to_attr_val = []

            for attr_val_index, attr_val in enumerate(unique_attribute_values):
                label_entropy_with_respect_to_this_attr_val = self.entropy_label_with_respect_to_attr_val(
                    attr_index, attr_val, feature_data, labels, unique_labels)
                label_entropies_with_respect_to_attr_val.append(label_entropy_with_respect_to_this_attr_val)

            # we have the attr probabilities and the entropies for all the unique values for this attribute
            # we have all needed to calculate the label entropy with respect to this attribute
            label_entropy_with_respect_to_this_attr = self.entropy_label_with_respect_to_attr(attr_val_probabilities,
                                                                   label_entropies_with_respect_to_attr_val)
            attr_entropies.append(label_entropy_with_respect_to_this_attr)

        gains = self.initial_label_entropy - attr_entropies

        max_gain_attr_index = np.argmax(gains)

        # split the data on the max_gain attribute
        data_split_by_attribute = self.split_data_by_attribute(feature_data, labels, max_gain_attr_index)

        print('---')

    @staticmethod
    def split_data_by_attribute(feature_data, labels, attr_index):

        unique_attribute_values, counts_per_value = np.unique(feature_data[:, attr_index], return_counts=True)
        data_split_by_attribute = []
        for attr_val_index, attr_val in enumerate(unique_attribute_values):
            record_indices_for_this_attr_val = np.where(feature_data[:, attr_index] == attr_val)
            feature_data_for_this_attr_val = feature_data[record_indices_for_this_attr_val]
            # remove the selected attribute
            feature_data_for_this_attr_val = np.delete(feature_data_for_this_attr_val, attr_index, 1)
            labels_for_this_attr_val = labels[record_indices_for_this_attr_val]
            unique_labels = np.unique(labels_for_this_attr_val)
            # check if leaf node
            if unique_labels.size == 1:
                print("Attribite: ", attr_index, ", Value: ", attr_val, ", is a leaf node with value: ", unique_labels[0])

            data_split_by_attribute.append((feature_data_for_this_attr_val, labels_for_this_attr_val))

        return data_split_by_attribute

    def entropy_label_with_respect_to_attr_val(self, attr_index, attr_val, feature_data, labels, unique_labels):

        record_indices_for_this_attr_val = np.where(feature_data[:, attr_index] == attr_val)
        per_label_counts_for_this_attr_value = []
        for unique_label in unique_labels:
            all_labels_this_attr_val = labels[record_indices_for_this_attr_val]
            num_records_for_this_attr_val_and_this_label = np.count_nonzero(all_labels_this_attr_val == unique_label)
            per_label_counts_for_this_attr_value.append(num_records_for_this_attr_val_and_this_label)

        return self.entropy(per_label_counts_for_this_attr_value)

    @staticmethod
    def entropy_label_with_respect_to_attr(attr_val_probabilities, label_entropies_with_respect_to_attr_val):

        return np.dot(attr_val_probabilities, np.array(label_entropies_with_respect_to_attr_val).T)

    @staticmethod
    def entropy(counts):

        probabilities = counts / np.sum(counts)
        log_probabilities = np.log2(probabilities)

        # log of 0 is negative infinity, but python gives us nan when we multiply with it, even if we multiply by zero
        # se we just change it to a really big negative number
        log_probabilities[log_probabilities == -np.inf] = -99999999999

        return np.dot(-probabilities, log_probabilities.T)



