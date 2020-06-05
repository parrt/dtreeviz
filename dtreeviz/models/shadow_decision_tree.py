from abc import ABC, abstractmethod
from typing import List, Tuple, Mapping
import numpy as np
from numbers import Number
from collections import Sequence
import pandas as pd
import sklearn
import xgboost

import dtreeviz



class ShadowDecTree3(ABC):
    def __init__(self,
                 tree_model,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.target_name = target_name
        self.class_names = class_names
        self.class_weight = self.get_class_weight()
        self.thresholds = self.get_thresholds()
        self.features = self.get_features()

        if not self.is_fit():
            raise Exception(f"Model {tree_model} is not fit.")

        self.x_data = ShadowDecTree3._get_x_data(x_data)
        self.y_data = ShadowDecTree3._get_y_data(y_data)
        self.node_to_samples = self.get_node_samples()
        if self.is_classifier():
            self.class_weights = self.get_class_weights()
            self.unique_target_values = np.unique(self.y_data)
        self.root, self.leaves, self.internal = self._get_tree_nodes()

        if class_names:
            self.class_names = self._get_class_names()

    @abstractmethod
    def is_fit(self):
        pass

    @abstractmethod
    def is_classifier(self):
        pass

    @abstractmethod
    def get_class_weights(self):
        pass

    @abstractmethod
    def get_thresholds(self):
        pass

    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def criterion(self):
        pass

    @abstractmethod
    def get_class_weight(self):
        pass

    @abstractmethod
    def nclasses(self):
        pass

    @abstractmethod
    def classes(self):
        pass

    @abstractmethod
    def get_node_samples(self):
        pass

    @abstractmethod
    def get_children_left(self):
        pass

    @abstractmethod
    def get_children_right(self):
        pass

    @abstractmethod
    def get_node_split(self, id) -> (int, float):
        pass

    @abstractmethod
    def get_node_feature(self, id) -> int:
        pass

    @abstractmethod
    def get_value(self, id):
        pass

    @abstractmethod
    def nnodes(self):
        pass

    @abstractmethod
    def get_node_criterion(self, id):
        pass

    @abstractmethod
    def get_feature_path_importance(self, node_list):
        pass

    @abstractmethod
    def get_max_depth(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_min_samples_leaf(self):
        pass

    def get_split_node_heights(self, X_train, y_train, nbins) -> Mapping[int, int]:
        # TODO
        # do we need to call np.unique(y_train) ? what about dtc.classes_
        class_values = np.unique(y_train)
        node_heights = {}
        # print(f"Goal {nbins} bins")
        for node in self.internal:
            # print(node.feature_name(), node.id)
            X_feature = X_train[:, node.feature()]
            overall_feature_range = (np.min(X_feature), np.max(X_feature))
            # print(f"range {overall_feature_range}")
            r = overall_feature_range[1] - overall_feature_range[0]

            bins = np.linspace(overall_feature_range[0],
                               overall_feature_range[1], nbins + 1)
            # bins = np.arange(overall_feature_range[0],
            #                  overall_feature_range[1] + binwidth, binwidth)
            # print(f"\tlen(bins)={len(bins):2d} bins={bins}")
            X, y = X_feature[node.samples()], y_train[node.samples()]
            X_hist = [X[y == cl] for cl in class_values]
            height_of_bins = np.zeros(nbins)
            for i, _ in enumerate(class_values):
                hist, foo = np.histogram(X_hist[i], bins=bins, range=overall_feature_range)
                # print(f"class {cl}: goal_n={len(bins):2d} n={len(hist):2d} {hist}")
                height_of_bins += hist
            node_heights[node.id] = np.max(height_of_bins)

            # print(f"\tmax={np.max(height_of_bins):2.0f}, heights={list(height_of_bins)}, {len(height_of_bins)} bins")
        return node_heights

    def predict(self, x: np.ndarray) -> Tuple[Number, List]:
        """
        Given an x-vector of features, return predicted class or value based upon
        this tree. Also return path from root to leaf as 2nd value in return tuple.
        Recursively walk down tree from root to appropriate leaf by
        comparing feature in x to node's split value. Also return

        :param x: Feature vector to run down the tree to a leaf.
        :type x: np.ndarray
        :return: Predicted class or value based
        :rtype: Number
        """

        def walk(t, x, path):
            if t is None:
                return None
            path.append(t)
            if t.isleaf():
                return t
            if x[t.feature()] < t.split():
                return walk(t.left, x, path)
            return walk(t.right, x, path)

        path = []
        leaf = walk(self.root, x, path)
        return leaf.prediction(), path

    def tesselation(self):
        """
        Walk tree and return list of tuples containing a leaf node and bounding box
        list of (x1,y1,x2,y2) coordinates
        :return:
        :rtype:
        """
        bboxes = []

        def walk(t, bbox):
            if t is None:
                return None
            # print(f"Node {t.id} bbox {bbox} {'   LEAF' if t.isleaf() else ''}")
            if t.isleaf():
                bboxes.append((t, bbox))
                return t
            # shrink bbox for left, right and recurse
            s = t.split()
            if t.feature() == 0:
                walk(t.left, (bbox[0], bbox[1], s, bbox[3]))
                walk(t.right, (s, bbox[1], bbox[2], bbox[3]))
            else:
                walk(t.left, (bbox[0], bbox[1], bbox[2], s))
                walk(t.right, (bbox[0], s, bbox[2], bbox[3]))

        # create bounding box in feature space (not zeroed)
        f1_values = self.x_data[:, 0]
        f2_values = self.x_data[:, 1]
        overall_bbox = (np.min(f1_values), np.min(f2_values),  # x,y of lower left edge
                        np.max(f1_values), np.max(f2_values))  # x,y of upper right edge
        walk(self.root, overall_bbox)

        return bboxes

    def get_leaf_sample_counts(self, min_samples=0, max_samples=None):
        """Get the number of samples for each leaf.

        There is the option to filter the leaves with less than min_samples or more than max_samples.

        :param min_samples: int
            Min number of samples for a leaf
        :param max_samples: int
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
        """Get the number of samples by class for each leaf.

        :return: tuple
            Contains a list of leaf ids and a two lists of leaf samples (one for each class)
        """

        leaf_samples = [(node.id, node.n_sample_classes()[0], node.n_sample_classes()[1]) for node in self.leaves]
        index, leaf_sample_0, leaf_samples_1 = zip(*leaf_samples)
        return index, leaf_sample_0, leaf_samples_1

    def _get_class_names(self):
        if self.nclasses() > 1:
            if isinstance(self.class_names, dict):
                # TODO does it make any sense ?
                return self.class_names
            elif isinstance(self.class_names, Sequence):
                return {i: n for i, n in enumerate(self.class_names)}
            else:
                raise Exception(f"class_names must be dict or sequence, not {self.class_names.__class__.__name__}")

    def _get_tree_nodes(self):

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = []  # non-leaf nodes
        children_left = self.get_children_left()
        children_right = self.get_children_right()

        def walk(node_id):
            if children_left[node_id] == -1 and children_right[node_id] == -1:  # leaf
                t = ShadowDecTreeNode(self, node_id)
                leaves.append(t)
                return t
            else:  # decision node
                left = walk(children_left[node_id])
                right = walk(children_right[node_id])
                t = ShadowDecTreeNode(self, node_id, left, right)
                internal.append(t)
                return t

        root_node_id = 0
        root = walk(root_node_id)
        return root, leaves, internal

    @staticmethod
    def _get_x_data(x_data):
        if isinstance(x_data, pd.DataFrame):
            x_data = x_data.values  # We recommend using :meth:`DataFrame.to_numpy` instead.
        return x_data

    @staticmethod
    def _get_y_data(y_data):
        if isinstance(y_data, pd.Series):
            y_data = y_data.values
        return y_data

    @staticmethod
    def get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names=None, tree_index=None):
        if isinstance(tree_model, ShadowDecTree3):
            return tree_model
        elif isinstance(tree_model, (sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier)):
            return dtreeviz.models.sklearn_decision_trees.SKDTree(tree_model, x_data, y_data, feature_names, target_name, class_names)
        elif isinstance(tree_model, xgboost.core.Booster):
            return dtreeviz.models.xgb_decision_tree.XGBDTree(tree_model, tree_index, x_data, y_data, feature_names, target_name, class_names)


class ShadowDecTreeNode():
    """
    A node in a shadow tree.  Each node has left and right
    pointers to child nodes, if any.  As part of tree construction process, the
    samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """

    def __init__(self, shadow_tree: ShadowDecTree3, id: int, left=None, right=None):
        self.shadow_tree = shadow_tree
        self.id = id
        self.left = left
        self.right = right

    def split(self) -> (int, float):
        return self.shadow_tree.get_node_split(self.id)

    def feature(self) -> int:
        return self.shadow_tree.get_node_feature(self.id)

    def feature_name(self) -> (str, None):
        if self.shadow_tree.feature_names is not None:
            return self.shadow_tree.feature_names[self.feature()]
        return None

    def samples(self) -> List[int]:
        return self.shadow_tree.node_to_samples[self.id]

    def nsamples(self) -> int:
        """
        Return the number of samples associated with this node. If this is a
        leaf node, it indicates the samples used to compute the predicted value
        or class. If this is an internal node, it is the number of samples used
        to compute the split point.
        """

        return len(self.samples())
        # return self.shadow_tree.dtree.get_node_nsamples(self.id)

    # TODO
    # rename method name
    def n_sample_classes(self):
        samples = np.array(self.samples())
        if samples.size == 0:
            return [0, 0]

        node_y_data = self.shadow_tree.y_data[samples]
        unique, counts = np.unique(node_y_data, return_counts=True)

        if len(unique) == 2:
            return [counts[0], counts[1]]
        elif len(unique) == 1:  # one node can contain samples from only on class
            if unique[0] == 0:
                return [counts[0], 0]
            elif unique[0] == 1:
                return [0, counts[0]]

    def criterion(self):
        return self.shadow_tree.get_node_criterion(self.id)

    def split_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the list of indexes to the left and the right of the split value.
        """
        samples = np.array(self.samples())
        node_X_data = self.shadow_tree.x_data[samples, self.feature()]
        split = self.split()
        left = np.nonzero(node_X_data < split)[0]
        right = np.nonzero(node_X_data >= split)[0]
        return left, right

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def isclassifier(self) -> bool:
        return self.shadow_tree.nclasses() > 1

    def prediction(self) -> (Number, None):

        if not self.isleaf():
            return None
        if self.isclassifier():
            counts = self.shadow_tree.get_value(self.id)
            return np.argmax(counts)
        else:
            return self.shadow_tree.get_value(self.id)[0]

    def prediction_name(self) -> (str, None):
        """
        If the tree model is a classifier and we know the class names,
        return the class name associated with the prediction for this leaf node.
        Return prediction class or value otherwise.
        """
        if self.isclassifier():
            if self.shadow_tree.class_names is not None:
                return self.shadow_tree.class_names[self.prediction()]
        return self.prediction()

    def class_counts(self) -> (List[int], None):
        """
        If this tree model is a classifier, return a list with the count
        associated with each class.
        """
        if self.isclassifier():
            if self.shadow_tree.class_weight is None:
                # return np.array(np.round(self.shadow_tree.tree_model.tree_.value[self.id][0]), dtype=int)
                return np.array(np.round(self.shadow_tree.get_value(self.id)), dtype=int)
            else:
                return np.round(
                    self.shadow_tree.get_value(self.id) / self.shadow_tree.class_weights).astype(int)
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "<pred={value},n={n}>".format(value=round(self.prediction(), 1), n=self.nsamples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(), 1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')
