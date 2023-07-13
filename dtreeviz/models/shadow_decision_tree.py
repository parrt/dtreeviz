from abc import ABC, abstractmethod
from numbers import Number
from typing import List, Tuple, Mapping

import numpy as np
import pandas as pd
import sklearn

from dtreeviz import utils


class ShadowDecTree(ABC):
    """
    This object adapts decision trees constructed by the various libraries such as scikit-learn's and XGBoost's
    DecisionTree(Regressor|Classifier) to dtreeviz.  As part of the construction process, the samples
    considered at decision and leaf nodes are saved as a big dictionary for use by the nodes.

    The decision trees for classifiers and regressors from scikit-learn and
    XGBoost etc... are built for efficiency, not ease of tree walking. This class
    wraps all of that information in an easy-to-use and consistent interface
    that hides the details of the various decision tree libraries.

    Field leaves is list of shadow leaf nodes.
    Field internal is list of shadow non-leaf nodes.
    Field root is the shadow tree root.
    """

    def __init__(self,
                 tree_model,
                 X_train: (pd.DataFrame, np.ndarray),
                 y_train: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):
        """
        Parameters
        ----------
        :param tree_model: sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier, xgboost.core.Booster
            The decision tree to be interpreted
        :param X_train: pd.DataFrame, np.ndarray
            Features values on which the shadow tree will be build.
        :param y_train: pd.Series, np.ndarray
            Target values on which the shadow tree will be build.
        :param feature_names: List[str]
            Features' names
        :param target_name: str
            Target's name
        :param class_names: List[str], Mapping[int, str]
            Class' names (in case of a classifier)

        """

        self.tree_model = tree_model
        if not self.is_fit():
            raise Exception(f"Model {tree_model} is not fit.")

        self.feature_names = feature_names
        self.target_name = target_name
        self.X_train = ShadowDecTree._get_x_data(X_train)
        self.y_train = ShadowDecTree._get_y_data(y_train)
        self.root, self.leaves, self.internal = self._get_tree_nodes()
        if self.is_classifier():
            self.class_names = utils._normalize_class_names(class_names, self.nclasses())

    @abstractmethod
    def is_fit(self) -> bool:
        """Checks if the tree model is already trained."""
        pass

    @abstractmethod
    def is_classifier(self) -> bool:
        """Checks if the tree model is a classifier."""
        pass

    @abstractmethod
    def get_class_weights(self):
        """Returns the tree model's class weights."""
        pass

    @abstractmethod
    def get_thresholds(self) -> np.ndarray:
        """Returns split node/threshold values for tree's nodes.

        Ex. threshold[i] holds the split value/threshold for the node i.
        """
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """Returns feature indexes for tree's nodes.

        Ex. features[i] holds the feature index to split on
        """
        pass

    @abstractmethod
    def criterion(self) -> str:
        """Returns the function to measure the quality of a split.

        Ex. Gini, entropy, MSE, MAE
        """
        pass

    @abstractmethod
    def get_class_weight(self):
        """
        TOOD - to be compared with get_class_weights
        :return:
        """
        pass

    @abstractmethod
    def nclasses(self) -> int:
        """Returns the number of classes.

        Ex. 2 for binary classification or 1 for regression.
        """
        pass

    @abstractmethod
    def classes(self) -> np.ndarray:
        """Returns the tree's classes values in case of classification.

        Ex. [0,1] in class of a binary classification
        """
        pass

    @abstractmethod
    def get_node_samples(self):
        """Returns dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        pass

    @abstractmethod
    def get_split_samples(self, id):
        """Returns left and right split indexes from a node"""
        pass

    @abstractmethod
    def get_node_nsamples(self, id):
        """Returns number of samples for a specific node id."""
        pass

    @abstractmethod
    def get_children_left(self) -> np.ndarray:
        """Returns the node ids of the left child node.

        Ex. children_left[i] holds the node id of the left child of node i.
        """
        pass

    @abstractmethod
    def get_children_right(self) -> np.ndarray:
        """Returns the node ids of the right child node.

        Ex. children_right[i] holds the node id of the right child of node i.
        """
        pass

    @abstractmethod
    def get_node_split(self, id) -> (int, float):
        """Returns node split value.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_node_feature(self, id) -> int:
        """Returns feature index from node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_node_nsamples_by_class(self, id):
        """For a classification decision tree, returns the number of samples for each class from a specified node.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_prediction(self, id):
        """Returns the constant prediction value for node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def nnodes(self) -> int:
        "Returns the number of nodes (internal nodes + leaves) in the tree."
        pass

    @abstractmethod
    def get_node_criterion(self, id):
        """Returns the impurity (i.e., the value of the splitting criterion) at node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_feature_path_importance(self, node_list):
        """Returns the feature importance for a list of nodes.

        The node feature importance is calculated based on only the nodes from that list, not based on entire tree nodes.

        Parameters
        ----------
        node_list : List
            The list of nodes.
        """
        pass

    @abstractmethod
    def get_max_depth(self) -> int:
        """The max depth of the tree."""
        pass

    @abstractmethod
    def get_score(self) -> float:
        """
        For classifier, returns the mean accuracy.
        For regressor, returns the R^2.
        """
        pass

    @abstractmethod
    def get_min_samples_leaf(self) -> (int, float):
        """Returns the minimum number of samples required to be at a leaf node, during node splitting"""
        pass

    @abstractmethod
    def shouldGoLeftAtSplit(self, id, x):
        """Return true if it should go to the left node child based on node split criterion and x value"""
        pass

    def get_root_edge_labels(self):
        pass

    def is_categorical_split(self, id) -> bool:
        """Checks if the node split is a categorical one.

        This method needs to be overloaded only for shadow tree implementation which contain categorical splits,
        like Spark.
        """
        return False

    def get_split_node_heights(self, X_train, y_train, nbins) -> Mapping[int, int]:
        class_values = np.unique(y_train)
        node_heights = {}
        for node in self.internal:
            # print(f"node feature {node.feature_name()}, id {node.id}")
            X_feature = X_train[:, node.feature()]
            if node.is_categorical_split():
                overall_feature_range = (0, len(np.unique(X_train[:, node.feature()])) - 1)
            else:
                overall_feature_range = (np.min(X_feature), np.max(X_feature))

            bins = np.linspace(overall_feature_range[0],
                               overall_feature_range[1], nbins + 1)
            X, y = X_feature[node.samples()], y_train[node.samples()]

            # in case there is a categorical split node, we can convert the values to numbers because we need them
            # only for getting the distribution values
            if node.is_categorical_split():
                X = pd.Series(X).astype("category").cat.codes

            X_hist = [X[y == cl] for cl in class_values]
            height_of_bins = np.zeros(nbins)
            for i, _ in enumerate(class_values):
                hist, foo = np.histogram(X_hist[i], bins=bins, range=overall_feature_range)
                height_of_bins += hist
            node_heights[node.id] = np.max(height_of_bins)
            # print(f"\tmax={np.max(height_of_bins):2.0f}, heights={list(height_of_bins)}, {len(height_of_bins)} bins")
        return node_heights

    def predict(self, x: np.ndarray) -> Number:
        """
        Given an x - vector of features, return predicted class or value based upon this tree.
        Also return path from root to leaf as 2nd value in return tuple.

        Recursively walk down tree from root to appropriate leaf by comparing feature in x to node's split value.

        :param
        x: np.ndarray
            Feature vector to run down the tree to a  leaf.
        """

        def walk(t, x):
            if t.isleaf():
                return t
            if self.shouldGoLeftAtSplit(t.id, x[t.feature()]):
                return walk(t.left, x)
            return walk(t.right, x)

        leaf = walk(self.root, x)
        return leaf.prediction()

    def predict_path(self, x: np.ndarray) -> List:
        """
        Given an x - vector of features, return path prediction based upon this tree.
        Also return path from root to leaf as 2nd value in return tuple.

        Recursively walk down tree from root to appropriate leaf by comparing feature in x to node's split value.

        :param
        x: np.ndarray
            Feature vector to run down the tree to a  leaf.
        """

        def walk(t, x, path):
            path.append(t)
            if t.isleaf():
                return None
            if self.shouldGoLeftAtSplit(t.id, x[t.feature()]):
                return walk(t.left, x, path)
            return walk(t.right, x, path)

        path = []
        walk(self.root, x, path)
        return path

    def get_leaf_sample_counts(self, min_samples=0, max_samples=None):
        """
        Get the number of samples for each leaf.

        There is the option to filter the leaves with samples between min_samples and max_samples.

        Parameters
        ----------
        min_samples: int
            Min number of samples for a leaf
        max_samples: int
            Max number of samples for a leaf

        :return: tuple
            Contains a numpy array of leaf ids and an array of leaf samples
        """
        max_samples = max_samples if max_samples else max([node.nsamples() for node in self.leaves])
        leaf_samples = [(node.id, node.nsamples()) for node in self.leaves if
                        min_samples <= node.nsamples() <= max_samples]
        x, y = zip(*leaf_samples)
        return np.array(x), np.array(y)

    def get_leaf_criterion(self):
        """Get criterion for each leaf

        For classification, supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        For regression, supported criteria are “mse”, “friedman_mse”, “mae”.
        """
        leaf_criterion = [(node.id, node.criterion()) for node in self.leaves]
        x, y = zip(*leaf_criterion)
        return np.array(x), np.array(y)

    def get_leaf_sample_counts_by_class(self):
        """ Get the number of samples by class for each leaf.

        :return: tuple
            Contains a list of leaf ids and other list with number of samples for each leaf
        """
        leaf_samples = [(node.id, self.get_node_nsamples_by_class(node.id)) for node in self.leaves]
        index, leaf_samples= zip(*leaf_samples)
        return index, leaf_samples

    def _get_tree_nodes(self):
        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = []  # non-leaf nodes
        children_left = self.get_children_left()
        children_right = self.get_children_right()

        def walk(node_id, level):
            if children_left[node_id] == -1 and children_right[node_id] == -1:  # leaf
                t = ShadowDecTreeNode(self, node_id, level=level)
                leaves.append(t)
                return t
            else:  # decision node
                left = walk(children_left[node_id], level + 1)
                right = walk(children_right[node_id], level + 1)
                t = ShadowDecTreeNode(self, node_id, left, right, level)
                internal.append(t)
                return t

        root_node_id = 0
        root = walk(root_node_id, 0)
        return root, leaves, internal

    @staticmethod
    def _get_x_data(X_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values  # We recommend using :meth:`DataFrame.to_numpy` instead.
        return X_train

    @staticmethod
    def _get_y_data(y_train):
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        return y_train

    @staticmethod
    def get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names=None, tree_index=None):
        """Get an internal representation of the tree obtained from a specific library"""
        # Sanity check
        if isinstance(X_train, pd.DataFrame):
            nancols = X_train.columns[X_train.isnull().any().values].tolist()
            if len(nancols)>0:
                raise ValueError(f"dtreeviz does not support NaN (see column(s) {', '.join(nancols)})")
        elif isinstance(X_train, np.ndarray):
            nancols = np.where(pd.isnull(X_train).any(axis=0))[0].astype(str).tolist()
            if len(nancols)>0:
                raise ValueError(f"dtreeviz does not support NaN (see column index(es) {', '.join(nancols)})")

        """
        To check to which library the tree_model belongs we are using string checks instead of isinstance()
        because we don't want all the libraries to be installed as mandatory, except sklearn.
        """
        if hasattr(tree_model, 'get_booster'):
            # scikit-learn wrappers XGBClassifier and XGBRegressor allow you to
            # extract the underlying xgboost.core.Booster with the get_booster() method:
            tree_model = tree_model.get_booster()
        if isinstance(tree_model, ShadowDecTree):
            return tree_model
        elif isinstance(tree_model, (sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier)):
            from dtreeviz.models import sklearn_decision_trees
            return sklearn_decision_trees.ShadowSKDTree(tree_model, X_train, y_train, feature_names,
                                                        target_name, class_names)
        elif str(type(tree_model)).endswith("xgboost.core.Booster'>"):
            from dtreeviz.models import xgb_decision_tree
            return xgb_decision_tree.ShadowXGBDTree(tree_model, tree_index, X_train, y_train,
                                                    feature_names, target_name, class_names)
        elif (str(type(tree_model)).endswith("pyspark.ml.classification.DecisionTreeClassificationModel'>") or
              str(type(tree_model)).endswith("pyspark.ml.regression.DecisionTreeRegressionModel'>")):
            from dtreeviz.models import spark_decision_tree
            return spark_decision_tree.ShadowSparkTree(tree_model, X_train, y_train,
                                                       feature_names, target_name, class_names)
        elif "lightgbm.basic.Booster" in str(type(tree_model)):
            from dtreeviz.models import lightgbm_decision_tree
            return lightgbm_decision_tree.ShadowLightGBMTree(tree_model, tree_index, X_train, y_train,
                                                             feature_names, target_name, class_names)
        elif any(tf_model in str(type(tree_model)) for tf_model in ["tensorflow_decision_forests.keras.RandomForestModel",
                                                                    "tensorflow_decision_forests.keras.GradientBoostedTreesModel"]):
            from dtreeviz.models import tensorflow_decision_tree
            return tensorflow_decision_tree.ShadowTensorflowTree(tree_model, tree_index, X_train, y_train,
                                                                 feature_names, target_name, class_names)
        else:
            raise ValueError(
                f"Tree model must be in (DecisionTreeRegressor, DecisionTreeClassifier, "
                "xgboost.core.Booster, lightgbm.basic.Booster, pyspark DecisionTreeClassificationModel, "
                f"pyspark DecisionTreeClassificationModel, tensorflow_decision_forests.keras.RandomForestModel, "
                f"tensorflow_decision_forests.keras.GradientBoostedTreesModel) "
                f"but you passed a {tree_model.__class__.__name__}!")


class ShadowDecTreeNode():
    """
    A node in a shadow tree. Each node has left and right pointers to child nodes, if any.
    As part of tree construction process, the samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """

    def __init__(self, shadow_tree: ShadowDecTree, id: int, left=None, right=None, level=None):
        self.shadow_tree = shadow_tree
        self.id = id
        self.left = left
        self.right = right
        self.level = level

    def split(self) -> (int, float):
        """Returns the split/threshold value used at this node."""
        return self.shadow_tree.get_node_split(self.id)

    def feature(self) -> int:
        """Returns feature index used at this node"""
        return self.shadow_tree.get_node_feature(self.id)

    def feature_name(self) -> (str, None):
        """Returns the feature name used at this node"""
        if self.shadow_tree.feature_names is not None:
            return self.shadow_tree.feature_names[self.feature()]
        return None

    def samples(self) -> List[int]:
        """Returns samples indexes from this node"""
        return self.shadow_tree.get_node_samples()[self.id]

    def nsamples(self) -> int:
        """
        Return the number of samples associated with this node. If this is a leaf node, it indicates the samples
        used to compute the predicted value or class . If this is an internal node, it is the number of samples used
        to compute the split point.
        """
        return self.shadow_tree.get_node_nsamples(self.id)

    def n_sample_classes(self):
        """Used for classification only.

        Returns the count values for each class.
        """

        samples = np.array(self.samples())
        node_values = [0] * len(self.shadow_tree.class_names)
        if samples.size == 0:
            return node_values
        node_y_data = self.shadow_tree.y_train[samples]
        unique, counts = np.unique(node_y_data, return_counts=True)

        for i in range(len(unique)):
            node_values[unique[i]] = counts[i]

        return node_values

    def criterion(self):
        return self.shadow_tree.get_node_criterion(self.id)

    def split_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the list of indexes to the left and the right of the split value."""
        return self.shadow_tree.get_split_samples(self.id)

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def isclassifier(self) -> bool:
        return self.shadow_tree.is_classifier()

    def is_categorical_split(self) -> bool:
        return self.shadow_tree.is_categorical_split(self.id)

    def prediction(self) -> (Number, None):
        """Returns leaf prediction.

        If the node is an internal node, returns None
        """
        if not self.isleaf():
            return None
        return self.shadow_tree.get_prediction(self.id)

    def prediction_name(self) -> (str, None):
        """
        If the tree model is a classifier and we know the class names, return the class name associated with the
        prediction for this leaf node.

        Return prediction class or value otherwise.
        """
        if self.isclassifier():
            # In a GBT model, the trees are always regressive trees (even if the GBT is a classifier).
            if "tensorflow_decision_forests.keras.GradientBoostedTreesModel" in str(type(self.shadow_tree.tree_model)):
                return round(self.prediction(), 6)
            if self.shadow_tree.class_names is not None:
                return self.shadow_tree.class_names[self.prediction()]
        return self.prediction()

    def class_counts(self) -> (List[int], None):
        """
        If this tree model is a classifier, return a list with the count associated with each class.
        """
        if self.isclassifier():
            if self.shadow_tree.get_class_weight() is None:
                return np.array(np.round(self.shadow_tree.get_node_nsamples_by_class(self.id)), dtype=int)
            else:
                return np.round(
                    self.shadow_tree.get_node_nsamples_by_class(self.id) / self.shadow_tree.get_class_weights()).astype(
                    int)
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "<pred={value},n={n}>".format(value=round(self.prediction(), 1), n=self.nsamples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(), 1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')


class VisualisationNotYetSupportedError(Exception):
    def __init__(self, method_name, model_name):
        super().__init__(f"{method_name} is not implemented yet for {model_name}. "
                         f"Please create an issue on https://github.com/parrt/dtreeviz/issues if you need this. Thanks!")
