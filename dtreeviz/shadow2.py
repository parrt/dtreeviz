from collections import defaultdict, Sequence
from numbers import Number
from typing import List, Tuple, Mapping

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost.core import Booster

from dtreeviz.models.decision_trees import SKDTree, XGBDTree


# TODO
# check again all properties and methods, done
# difference between class_weight and class_weights
# we need to find a way to initialize more easily the ShadowDecTree (ex. None)
# add docs
# ask Terence to look over all visualisations
# interpretation.py is not yet adapted
class ShadowDecTree2:
    def __init__(self, tree_model,
                 X_train=None,
                 y_train=None,
                 feature_names: List[str] = None,
                 class_names: (List[str], Mapping[int, str]) = None,
                 tree_index: int = 0):
        self.feature_names = feature_names
        self.class_names = class_names
        self.dtree = self._get_dtree_type(tree_model, tree_index)
        self.class_weight = self.dtree.get_class_weight()

        if not self.dtree.is_fit():
            raise Exception(f"Model {tree_model} is not fit.")

        if class_names:
            self.class_names = ShadowDecTree2._get_class_names(self.dtree, self.class_names)

        if X_train is not None and y_train is not None:
            self.X_train = ShadowDecTree2._get_x_train(X_train)
            self.y_train = ShadowDecTree2._get_y_train(y_train)
            self.class_weights = self.dtree.get_class_weights(y_train)
            self.unique_target_values = np.unique(y_train)
            self.node_to_samples = self.dtree.get_node_samples(self.X_train)

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = []  # non-leaf nodes
        children_left = self.dtree.get_children_left()
        children_right = self.dtree.get_children_right()

        print(children_left)
        print(children_right)

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
        # record root to actual shadow nodes
        self.root = walk(root_node_id)
        self.leaves = leaves
        self.internal = internal

    def nclasses(self):
        return self.dtree.get_n_classes()

    def nnodes(self):
        return self.dtree.get_node_count()

    # def leaf_sample_counts(self) -> List[int]:
    #     # TODO
    #     # should we send leaves, internal structure from shadow.py ?
    #     return self.dtree.get_leaf_sample_counts(self.leaves)

    def isclassifier(self):
        return self.nclasses() > 1

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
        f1_values = self.X_train[:, 0]
        f2_values = self.X_train[:, 1]
        overall_bbox = (np.min(f1_values), np.min(f2_values),  # x,y of lower left edge
                        np.max(f1_values), np.max(f2_values))  # x,y of upper right edge
        walk(self.root, overall_bbox)

        return bboxes

    # TO be implemented
    # @staticmethod
    # def node_samples(tree_model, data) -> Mapping[int, list]:
    #     """
    #     Return dictionary mapping node id to list of sample indexes considered by
    #     the feature/split decision.
    #     """
    #     # Doc say: "Return a node indicator matrix where non zero elements
    #     #           indicates that the samples goes through the nodes."
    #     dec_paths = tree_model.decision_path(data)
    #
    #     # each sample has path taken down tree
    #     node_to_samples = defaultdict(list)
    #     for sample_i, dec in enumerate(dec_paths):
    #         _, nz_nodes = dec.nonzero()
    #         for node_id in nz_nodes:
    #             node_to_samples[node_id].append(sample_i)
    #
    #     return node_to_samples

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

        leaf_samples = [(node.id, node.prediction_value()[0][0], node.prediction_value()[0][1]) for node in self.leaves]

        index, leaf_samples_0, leaf_samples_1 = zip(*leaf_samples)
        return index, leaf_samples_0, leaf_samples_1

    def _get_dtree_type(self, tree_model, tree_index=None):
        # factory method for dtree
        if isinstance(tree_model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            dtree = SKDTree(tree_model)
        elif isinstance(tree_model, Booster):
            dtree = XGBDTree(tree_model, tree_index)
        else:
            raise Exception("raise unknown dtree type")

        return dtree

    @staticmethod
    def _get_class_names(dtree, class_names):
        if dtree.get_n_classes() > 1:
            if isinstance(class_names, dict):
                # TODO does it make any sense ?
                return class_names
            elif isinstance(class_names, Sequence):
                return {i: n for i, n in enumerate(class_names)}
            else:
                raise Exception(f"class_names must be dict or sequence, not {class_names.__class__.__name__}")

    @staticmethod
    def _get_x_train(X_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values  # We recommend using :meth:`DataFrame.to_numpy` instead.
        return X_train

    @staticmethod
    def _get_y_train(y_train):
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        return y_train


class ShadowDecTreeNode():
    """
    A node in a shadow tree.  Each node has left and right
    pointers to child nodes, if any.  As part of tree construction process, the
    samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """

    def __init__(self, shadow_tree: ShadowDecTree2, id: int, left=None, right=None):
        self.shadow_tree = shadow_tree
        self.id = id
        self.left = left
        self.right = right

    def split(self) -> (int, float):
        return self.shadow_tree.dtree.get_node_split(self.id)

    def feature(self) -> int:
        return self.shadow_tree.dtree.get_node_feature(self.id)

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
        return self.shadow_tree.dtree.get_node_nsamples(self.id)

    def criterion(self):
        return self.shadow_tree.dtree.get_node_criterion(self.id)

    def split_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the list of indexes to the left and the right of the split value.
        """
        samples = np.array(self.samples())
        node_X_data = self.shadow_tree.X_train[samples, self.feature()]
        split = self.split()
        left = np.nonzero(node_X_data < split)[0]
        right = np.nonzero(node_X_data >= split)[0]
        return left, right

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def isclassifier(self) -> bool:
        return self.shadow_tree.dtree.get_n_classes() > 1

    def prediction(self) -> (Number, None):

        if not self.isleaf():
            return None
        if self.isclassifier():
            counts = self.shadow_tree.dtree.get_value(self.id)
            return np.argmax(counts)
        else:
            return self.shadow_tree.dtree.get_value(self.id)[0]

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

    def prediction_value(self):
        return self.shadow_tree.dtree.get_prediction_value(self.id)

    def class_counts(self) -> (List[int], None):
        """
        If this tree model is a classifier, return a list with the count
        associated with each class.
        """
        if self.isclassifier():
            if self.shadow_tree.class_weight is None:
                # return np.array(np.round(self.shadow_tree.tree_model.tree_.value[self.id][0]), dtype=int)
                return np.array(np.round(self.shadow_tree.dtree.get_value(self.id)), dtype=int)
            else:
                return np.round(
                    self.shadow_tree.dtree.get_value(self.id) / self.shadow_tree.class_weights).astype(int)
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "<pred={value},n={n}>".format(value=round(self.prediction(), 1), n=self.nsamples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(), 1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')
