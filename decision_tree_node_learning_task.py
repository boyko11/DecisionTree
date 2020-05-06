class DecisionTreeNodeLearningTask:

    def __init__(self):
        self.parent_id = None
        self.attributes_to_filter = []
        self.parent_split_attribute = None
        self.parent_split_value = None