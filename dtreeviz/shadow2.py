from collections import Sequence
from numbers import Number
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dtreeviz.models.trees import SKDTree

# TODO
# check again all properties and methods
# difference between class_weight and class_weights
class ShadowDecTree2:
    def __init__(self, tree_model,
                 X_train,
                 y_train,
                 feature_names,
                 class_names):
        self.feature_names = feature_names
        self.class_names = class_names
        self.dtree = self._get_dtree_type(tree_model)

        if not self.dtree.is_fit():
            raise Exception(f"Model {tree_model} is not fit.")

        self.class_names = ShadowDecTree2._get_class_names(self.dtree, self.class_names)
        self.X_train = ShadowDecTree2._get_x_train(X_train)
        self.y_train = ShadowDecTree2._get_y_train(y_train)
        self.class_weight = tree_model.class_weight
        self.class_weights = self.dtree.get_class_weight(y_train)
        self.node_to_samples = self.dtree.get_node_samples(self.X_train)

        # difference between local variable and self. variables
        children_left = self.dtree.get_children_left()
        children_right = self.dtree.get_children_right()

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = []  # non-leaf nodes

        def walk(node_id):
            if (children_left[node_id] == -1 and children_right[node_id] == -1):  # leaf
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

    @staticmethod
    def _get_dtree_type(tree_model):
        # factory method for dtree
        if isinstance(tree_model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            dtree = SKDTree(tree_model)
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
        return len(self.samples())

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
