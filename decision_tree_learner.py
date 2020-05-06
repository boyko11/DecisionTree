import numpy as np
from decision_tree_node_learning_task import DecisionTreeNodeLearningTask
from decision_tree_node import DecisionTreeNode


class DecisionTreeLearner:

    def __init__(self):

        # the tree is a table, implemented here as a list of DecisionTreeNodes
        self.decision_tree = []
        self.label_probabilities = {}
        self.initial_label_entropy_with_respect_to_features = None

        self.feature_data = None
        self.labels = None
        self.node_id_sequence = 0

        self.node_learning_queue = []

    def get_node_id_sequence(self):
        self.node_id_sequence += 1
        return self.node_id_sequence - 1

    def train(self, feature_data, labels):

        # store the data instead of passing it around
        self.feature_data = feature_data
        self.labels = labels

        # start with learning the root node, then queue up learning for next levels of the tree
        root_node_learning_task = DecisionTreeNodeLearningTask()
        # id of 0 signifies the root
        root_node_learning_task.node_id = self.get_node_id_sequence()
        root_node_learning_task.parent_id = None
        root_node_learning_task.attributes_to_filter = []
        root_node_learning_task.parent_split_attribute = None
        root_node_learning_task.parent_split_value = None

        self.node_learning_queue.append(root_node_learning_task)

        while len(self.node_learning_queue) > 0:
            node_learning_task = self.node_learning_queue.pop(0)
            self.train_node(node_learning_task)

        print("Learned me a tree.")

    def filter_training_data_for_tree_node(self, node_learning_task):

        feature_data = self.feature_data
        labels = self.labels

        # get the data only for the attribute split_value
        if node_learning_task.parent_split_attribute is not None:
            record_indices_for_this_attr_val = np.where(
                feature_data[:, node_learning_task.parent_split_attribute] == node_learning_task.parent_split_value)[0]
            feature_data = feature_data[record_indices_for_this_attr_val, :]
            labels = self.labels[record_indices_for_this_attr_val]

        # remove the already used attributes
        if len(node_learning_task.attributes_to_filter) > 0:
            feature_data = np.delete(feature_data, node_learning_task.attributes_to_filter, axis=1)

        return feature_data, labels

    def train_node(self, node_learning_task):

        # create node
        this_tree_node = DecisionTreeNode()
        this_tree_node.id = node_learning_task.node_id
        this_tree_node.parent_id = node_learning_task.parent_id
        this_tree_node.split_attribute_index = None
        this_tree_node.parent_split_value = node_learning_task.parent_split_value
        this_tree_node.answer = None
        this_tree_node.filtered_attributes = node_learning_task.attributes_to_filter

        feature_data, labels = self.filter_training_data_for_tree_node(node_learning_task)

        # check if leaf

        #   if only one unique label - then, yes, I'm a leaf and the answer is the label
        unique_labels, counts_per_label = np.unique(labels, return_counts=True)
        if unique_labels.size == 1:
            print("Attribite: ", node_learning_task.parent_split_attribute,
                  ", Value: ", node_learning_task.parent_split_value, ", is a leaf node with answer: ", unique_labels[0])
            this_tree_node.answer = unique_labels[0]
            this_tree_node.leaf = True
            self.decision_tree.append(this_tree_node)

            return

        #   if only one attribute left - then, yes, I'm a leaf and the answer is the majority label
        if feature_data.ndim == 1 or feature_data.shape[1] == 1:
            majority_label_index = np.argmax(counts_per_label)
            this_tree_node.answer = unique_labels[majority_label_index]
            this_tree_node.leaf = True
            self.decision_tree.append(this_tree_node)
            print("Run out of attributes leaf: ", unique_labels, counts_per_label, this_tree_node.answer)

            return

        # Well, I'm not a leaf, so I have to figure out which attribute to split on
        split_attribute_index = self.calculate_split_attribute(self.feature_data,
                                                           node_learning_task.attributes_to_filter, self.labels, labels)

        this_tree_node.split_attribute_index = split_attribute_index
        self.decision_tree.append(this_tree_node)

        # for each unique attribute value for this index, queue up a node_learning task
        unique_attribute_values = np.unique(self.feature_data[:, split_attribute_index])
        for attr_val in unique_attribute_values:
            child_node_learning_task = DecisionTreeNodeLearningTask()
            # id of 0 signifies the root
            child_node_learning_task.node_id = self.get_node_id_sequence()
            child_node_learning_task.parent_id = this_tree_node.id
            child_node_learning_task.attributes_to_filter = this_tree_node.filtered_attributes.copy()
            child_node_learning_task.attributes_to_filter.append(split_attribute_index)
            child_node_learning_task.parent_split_attribute = split_attribute_index
            child_node_learning_task.parent_split_value = attr_val
            self.node_learning_queue.append(child_node_learning_task)

        print('---')

    def calculate_split_attribute(self, feature_data, filtered_attributes, labels, filtered_labels):

        # calculate probability of labels
        unique_labels, counts_per_label = np.unique(filtered_labels, return_counts=True)
        initial_label_entropy = self.entropy(counts_per_label)

        # for each feature
        attr_entropies = []
        attr_indices = []
        for attr_index in range(feature_data.shape[1]):

            if attr_index in filtered_attributes:
                continue

            attr_indices.append(attr_index)

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

        gains = initial_label_entropy - attr_entropies

        max_gain_attr_index = np.argmax(gains)

        return attr_indices[max_gain_attr_index]

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

    def classify(self, record):

        # traverse the tree down until you find a leaf
        current_node = self.decision_tree[0]

        while not current_node.leaf:
            split_attr = current_node.split_attribute_index
            split_attr_val = record[split_attr]
            current_node = self.fetch_child_by_parent_id_and_split_value(current_node.id, split_attr_val)

        return current_node.answer

    def fetch_child_by_parent_id_and_split_value(self, parent_id, split_attr_val):

        # this can be made more efficient
        return [node for node in self.decision_tree if (node.parent_id == parent_id
                                                        and node.parent_split_value == split_attr_val)][0]



