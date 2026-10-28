from Node import Tree_Node

import numpy as np

def compute_entropy(y, indices):
    a = y[indices]
    entropy = 0 
    for cl in np.unique(a):
        prob_cl = sum(a == cl) / len(a)

        entropy -= prob_cl * np.log2(prob_cl)
    return entropy

def compute_value(y, indices):
    a = y[indices]
    label, counts = np.unique(a, return_counts=True)
    return label[np.argmax(counts)]

def threshold_split(X, y, feature_index, indices):
    """Return threshold (or label for classification prob)"""

def feature_index_split(X, y, feature_subset, indices):
    feature_index = None
    threshold = None
    left_indices = None
    right_indices = None

    """Return feature index, threshold we choose to split data"""
    ...
    return feature_index, threshold, left_indices, right_indices

class Decision_Tree:
    def __init__(self, min_sample_split, max_depth, min_impurity_decrease, feature_subset = None):
        self.feature_subset = feature_subset
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X: np.ndarray, y: np.ndarray):
        def _grow_tree(node, depth):
            num_samples = len(node.indices)
            labels = y[node.indices]
            if (depth > self.max_depth or num_samples < self.min_sample_split or len(np.unique(labels)) == 1):
                node.value = compute_value(y, node.indices)
                return 
            node.feature_index, node.threshold, left_indices, right_indices = feature_index_split(X, y, self.feature_subset, node.indices)
            if left_indices is None or right_indices is None:
                node.value = compute_value(y, node.indices)
                return
            node.left = Tree_Node(left_indices)
            node.right = Tree_Node(right_indices)
            _grow_tree(node.left, depth + 1)
            _grow_tree(node.right, depth + 1)

        self.root = Tree_Node(list(range(X.shape[0])))
        _grow_tree(self.root, 0)
        return self