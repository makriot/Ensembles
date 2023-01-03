import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from time import time


class RandomForestMSE:
    """
    Implementation of simple Random Forest, contains:
    Methods: 
        fit(X, y, X_val, y_val) -- train model
        predict(X) -- create a predictions
    Attributes:
        scores_ -- MSE scores on each tree addition
        time_ -- training time after each tree addition
    """
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.estimators = []
        self.scores_ = []
        self.time_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        history = False
        if X_val is not None and y_val is not None:
            history = True
            predictions = []
        train_predictions = []
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[-1]//3
        base_time = time()
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_features=self.feature_subsample_size,
                                         max_depth=self.max_depth,
                                         **self.trees_parameters)
            train_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[train_indices], y[train_indices])
            train_predictions.append(tree.predict(X))
            self.estimators.append(tree)
            self.time_.append(time() - base_time)
            if history:
                predictions.append(tree.predict(X_val))
                self.scores_.append(np.sum((np.array(y_val) - np.mean(predictions, axis=0))**2)/y_val.size)
                yield np.sum((np.array(y) - np.mean(train_predictions, axis=0))**2)/y.size, self.scores_[-1]
            else:
                yield np.sum((np.array(y) - np.mean(train_predictions, axis=0))**2)/y.size, ""

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = []
        for tree in self.estimators:
            predictions.append(tree.predict(X))
        return np.mean(predictions, axis=0)


class GradientBoostingMSE:
    """
    Implementation of simple Gradient Boosting, contains:
    Methods: 
        fit(X, y, X_val, y_val) -- train model
        predict(X) -- create a predictions
    Attributes:
        self.weights_ -- weights of models
        scores_ -- MSE scores on each tree addition
        time_ -- training time after each tree addition
    """
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.estimators = []
        self.weights_ = []
        self.scores_ = []
        self.time_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        history = False
        if X_val is not None and y_val is not None:
            history = True
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[-1]//3
        base_time = time()

        ### init start
        tree = DecisionTreeRegressor(max_features=self.feature_subsample_size,
                                         max_depth=self.max_depth,
                                         **self.trees_parameters)
        base_time = time()
        tree.fit(X, y)
        self.time_.append(time() - base_time)
        y_pred = tree.predict(X)
        base_prediction = y_pred
        s = y - y_pred
        self.estimators.append(tree)
        self.weights_.append(1)
        if history:
            base_test_prediction = tree.predict(X_val)
            self.scores_.append(np.sum((y_val - base_test_prediction)**2))
        if history:
            yield np.sum(s**2)/y.size, self.scores_[-1]
        else:
            yield np.sum(s**2)/y.size, ""
        ### init finish

        for _ in range(self.n_estimators - 1):
            tree = DecisionTreeRegressor(max_features=self.feature_subsample_size,
                                         max_depth=self.max_depth,
                                         **self.trees_parameters)
            tree.fit(X, s)
            y_pred = tree.predict(X)
            w = minimize_scalar(lambda w: np.sum((base_prediction + w * y_pred - y)**2)).x
            base_prediction += self.learning_rate * w * y_pred
            s = y - base_prediction
            self.weights_.append(self.learning_rate * w)
            self.estimators.append(tree)
            if history:
                base_test_prediction += tree.predict(X_val) * self.learning_rate * w
                self.scores_.append(np.sum((y_val - base_test_prediction)**2)/y_val.size)

            self.time_.append(time() - base_time)

            if history:
                yield np.sum(s**2)/y.size, self.scores_[-1]
            else:
                yield np.sum(s**2)/y.size, ""

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction = np.zeros(X.shape[0])
        for w, tree in zip(self.weights_, self.estimators):
            prediction += w * tree.predict(X)
        return prediction
