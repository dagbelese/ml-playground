# Import libraries
import numpy as np

class Node():
    
    def __init__(self, feature_index=None, threshold=None, info_gain=None ,left=None, right=None, value=None):
        """
        Initialize Node.

        Args:
            feature_index (int): Index of the feature used for splitting.
            threshold (float): Threshold value for splitting the data.
            info_gain (float): Information gain achieved by the split.
            left (Node): Left child node.
            right (Node): Right child node.
            value (float/None): Leaf node value if it is a terminal node.
        """
        # Decision Node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # Leaf Node
        self.value = value


class DecisionTreeClassifier():
    """
    Implementation of Decision Tree Classifier.
    """
    def __init__(self, criterion='gini', max_depth=2, min_samples_split=2):
        """
        Initialize Decision Tree Classifier.

        Args:
            criterion (str): Splitting criterion, either 'gini' or 'entropy'.
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split a node.
        """
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, dataset, curr_depth=0):
        """
        Recursively builds the decision tree.
    
        Args:
            dataset (numpy.ndarray): Dataset including features and target values.
            curr_depth (int): Current depth of the tree during recursive building.
        Returns:
            Node: Root node of the constructed decision tree.
        """
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)

                return Node(best_split['feature_index'],
                            best_split['threshold'],
                            best_split['info_gain'],
                            left_subtree,
                            right_subtree)

        return Node(value=self.calculate_leaf_value(Y))

    def get_best_split(self, dataset, num_samples, num_features):
        """
        Identifies the best split based on the highest information gain.
    
        Args:
            dataset (numpy.ndarray): Dataset to be split.
            num_samples (int): Number of samples in the dataset.
            num_features (int): Number of features in the dataset.

        Returns:
            dict: Dictionary containing details about the best split.
        """
        best_split = {}
        max_info_gain = -np.inf

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, y_left, y_right = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, y_left, y_right)
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        """
        Splits the dataset into left and right based on the threshold value.
    
        Args:
            dataset (numpy.ndarray): Dataset to be split.
            feature_index (int): Index of the feature to split on.
            threshold (float): Threshold value for the split.

        Returns:
            tuple: Left and right subsets of the dataset.
        """
        # Efficiently splits dataset using index masks.
        left_indices = np.where(dataset[:, feature_index] <= threshold)[0]
        right_indices = np.where(dataset[:, feature_index] > threshold)[0]

        return dataset[left_indices], dataset[right_indices]

    def information_gain(self, parent, left_child, right_child):
        """
        Computes the information gain of a split.
    
        Args:
            parent (numpy.ndarray): Parent node class labels.
            left_child (numpy.ndarray): Left child node class labels.
            right_child (numpy.ndarray): Right child node class labels.

        Returns:
            float: Information gain of the split.
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        if self.criterion == 'gini':
            info_gain = self.gini_index(parent) - ((weight_left * self.gini_index(left_child)) + (weight_right * self.gini_index(right_child)))
        elif self.criterion == 'entropy':
            info_gain = self.entropy(parent) - ((weight_left * self.entropy(left_child)) + (weight_right * self.entropy(right_child)))
        
        return info_gain
    
    def gini_index(self, y):
        """
        Computes the Gini Index for a set of class labels.
    
        Args:
            y (numpy.ndarray): Array of class labels.

        Returns:
            float: Gini Index value.
        """
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y) 
        gini_index = 1 - np.sum(probs ** 2)
        
        return gini_index


    def entropy(self, y):
        """
        Computes the Entropy for a set of class labels.
    
        Args:
            y (numpy.ndarray): Array of class labels.

        Returns:
            float: Entropy value.
        """
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y) 
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def calculate_leaf_value(self, Y):
        """
        Compute the most common class label in the leaf node.
    
        Args:
            Y (numpy.ndarray): Array of class labels in the node.
        
        Returns:
            int: The most frequent class label.
        """
        return np.bincount(Y.astype(int)).argmax()

    def fit(self, X, Y):
        """
        Train Decision Tree Classifier.
    
        Args:
            X (numpy.ndarray): Training data of shape (num_samples, num_features).
            Y (numpy.ndarray): Target values of shape (num_samples,).
        
        Returns:
            self: Trained model.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be NumPy arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")
        
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Predict class labels for input data.
    
        Args:
            X (numpy.ndarray): Input data of shape (num_samples, num_features).
        
        Returns:
            numpy.ndarray: Predicted class labels.
        """
        # Use numpy apply_along_axis for faster predictions. 
        predictions = np.apply_along_axis(self.make_prediction, axis=1, arr=X, tree=self.root)
        
        return predictions
    
    def make_prediction(self, x, tree):
        """
        Iteratively predict the class label for a single data point.
    
        Args:
            x (numpy.ndarray): Single data point.
            tree (Node): The root node of the decision tree.
        
        Returns:
            int: Predicted class label for the input data point.
        """
        while tree.value is None:
            tree = tree.left if x[tree.feature_index] <= tree.threshold else tree.right
        
        return tree.value