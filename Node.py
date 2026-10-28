class Tree_Node:
    def __init__(self, indices = [], feature_index = None, threshold = None, left = None, right = None, value = None):
        self.indices = indices
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    