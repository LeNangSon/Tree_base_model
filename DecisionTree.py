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

def feature_index_split(X, y, feature_subset, indices):
    feature_index = None
    threshold_split = None
    left_indices = None
    right_indices = None

    """Return feature index, threshold we choose to split data"""
    min_impurity = 1e9

    # O(nm) n: cardinality of feature_subset, m: average number of unique values in each feature
    # One vs rest 
    for feature in feature_subset:
        for threshold in np.unique(X[:, feature]):
            tmp_left_indices = np.where(X[:, feature] == threshold)[0]
            tmp_right_indices = np.where(X[:, feature] != threshold)[0]
            left_node_impurity = compute_entropy(y, left_indices)
            right_node_impurity = compute_entropy(y, right_indices)
            sum_impurity = left_node_impurity + right_node_impurity
            if (sum_impurity < min_impurity):
                min_impurity = sum_impurity
                feature_index = feature
                threshold_split = threshold
                left_indices = tmp_left_indices
                right_indices = tmp_right_indices
    return feature_index, threshold_split, left_indices, right_indices

class Decision_Tree:
    def __init__(self, min_sample_split, max_depth, min_impurity_decrease, feature_subset = None):
        """
            feature_subset: [0,1,2,3,...]
        """
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