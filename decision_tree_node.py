class DecisionTreeNode:

    def __init__(self):
        self.id = None
        self.parent_id = None
        self.leaf = False
        self.split_attribute_index = None
        self.parent_split_value = None
        self.answer = None
        self.filtered_attributes = None