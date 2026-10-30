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
    best_gain = -1 
    best_thresh = None 
    best_left = None 
    best_right = None 
    current_entropy = compute_entropy(y , indices)
    n = len(indices) 
    #using feature_val as thresh to split 
    feature_vals = X[indices, feature_index]
    unique_thresholds = np.unique(feature_vals)

    for thresh in unique_thresholds:
        left_mask = feature_vals <= thresh
        right_mask = ~left_mask #opposite

        left_idx = [indices[i] for i in range(n) if left_mask[i]]
        right_idx = [indices[i] for i in range(n) if right_mask[i]]
    
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue        
        gain = current_entropy - (
            (len(left_idx) / n) * compute_entropy(y, left_idx) +
            (len(right_idx) / n) * compute_entropy(y, right_idx)
        )

        if gain > best_gain:
            best_gain = gain
            best_thresh = thresh
            best_left = left_idx
            best_right = right_idx

    return best_thresh, best_gain, best_left, best_right



def feature_index_split(X, y, feature_subset, indices):
    feature_index = None
    threshold = None
    left_indices = None
    right_indices = None
    max_gain = -1
    """Return feature index, threshold we choose to split data"""
    #avoid feature subnet is None , so loop is inf 
    if feature_subset is None:
        feature_subset = range(X.shape[1])

    for feat_idx in feature_subset:
        thresh, gain, left_idx, right_idx = threshold_split(X, y, feat_idx, indices)
        
        if gain is not None and gain > max_gain:
            max_gain = gain
            feature_index = feat_idx
            threshold = thresh
            left_indices = left_idx
            right_indices = right_idx

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
                node.indices = None 
                return 
            node.feature_index, node.threshold, left_indices, right_indices = feature_index_split(X, y, self.feature_subset, node.indices)
            if left_indices is None or right_indices is None:
                node.value = compute_value(y, node.indices)
                node.indices = None 
                return
            node.left = Tree_Node(left_indices)
            node.right = Tree_Node(right_indices)
            _grow_tree(node.left, depth + 1)
            _grow_tree(node.right, depth + 1)
            node.indices = None 
        self.root = Tree_Node(list(range(X.shape[0])))
        _grow_tree(self.root, 0) 
        return self

    #add predict to test (gemini 3.8 flash)
    def predict(self , X : np.ndarray):
        return np.array([self._predict_row(row, self.root) for row in X])
    def _predict_row(self, row, node):       
        if node.value is not None:
            return node.value
            # Đi theo nhánh tương ứng
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)