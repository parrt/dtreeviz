import numpy as np
import pandas as pd
from collections import defaultdict, Sequence
from typing import Mapping, List, Tuple
from numbers import Number
from sklearn.utils import compute_class_weight


class ShadowDecTree:
    """
    The decision trees for classifiers and regressors from scikit-learn
    are built for efficiency, not ease of tree walking. This class
    is intended as a way to wrap all of that information in an easy to use
    package.

    This tree shadows a decision tree as constructed by scikit-learn's
    DecisionTree(Regressor|Classifier).  As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved as a big dictionary for use by the nodes.

    Field leaves is list of shadow leaf nodes. Field internal is list of
    shadow non-leaf nodes.

    Field root is the shadow tree root.

    Parameters
    ----------
    class_names : (List[str],Mapping[int,str]). A mapping from target value
                  to target class name. If you pass in a list of strings,
                  target value i must be associated with class name[i]. You
                  can also pass in a dict that maps value to name.
    """
    def __init__(self, tree_model,
                 X_train,
                 y_train,
                 feature_names : List[str],
                 class_names : (List[str],Mapping[int,str])=None):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.class_weight = tree_model.class_weight

        if getattr(tree_model, 'tree_') is None: # make sure model is fit
            tree_model.fit(X_train, y_train)

        if tree_model.tree_.n_classes > 1:
            if isinstance(self.class_names, dict):
                self.class_names = self.class_names
            elif isinstance(self.class_names, Sequence):
                self.class_names = {i:n for i, n in enumerate(self.class_names)}
            else:
                raise Exception(f"class_names must be dict or sequence, not {self.class_names.__class__.__name__}")

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        self.X_train = X_train
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        self.y_train = y_train
        self.node_to_samples = ShadowDecTree.node_samples(tree_model, X_train)
        if self.isclassifier():
            self.unique_target_values = np.unique(y_train)
            self.class_weights = compute_class_weight(tree_model.class_weight, self.unique_target_values, self.y_train)

        tree = tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = [] # non-leaf nodes

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
        return self.tree_model.tree_.n_classes[0]

    def nnodes(self) -> int:
        "Return total nodes in the tree"
        return self.tree_model.tree_.node_count

    def leaf_sample_counts(self) -> List[int]:
        return [self.tree_model.tree_.n_node_samples[leaf.id] for leaf in self.leaves]

    def isclassifier(self):
        return self.tree_model.tree_.n_classes > 1

    def get_split_node_heights(self, X_train, y_train, nbins) -> Mapping[int,int]:
        class_values = self.unique_target_values
        node_heights = {}
        # print(f"Goal {nbins} bins")
        for node in self.internal:
            # print(node.feature_name(), node.id)
            X_feature = X_train[:, node.feature()]
            overall_feature_range = (np.min(X_feature), np.max(X_feature))
            # print(f"range {overall_feature_range}")
            r = overall_feature_range[1] - overall_feature_range[0]

            bins = np.linspace(overall_feature_range[0],
                               overall_feature_range[1], nbins+1)
            # bins = np.arange(overall_feature_range[0],
            #                  overall_feature_range[1] + binwidth, binwidth)
            # print(f"\tlen(bins)={len(bins):2d} bins={bins}")
            X, y = X_feature[node.samples()], y_train[node.samples()]
            X_hist = [X[y == cl] for cl in class_values]
            height_of_bins = np.zeros(nbins)
            for i,_ in enumerate(class_values):
                hist, foo = np.histogram(X_hist[i], bins=bins, range=overall_feature_range)
                # print(f"class {cl}: goal_n={len(bins):2d} n={len(hist):2d} {hist}")
                height_of_bins += hist
            node_heights[node.id] = np.max(height_of_bins)

            # print(f"\tmax={np.max(height_of_bins):2.0f}, heights={list(height_of_bins)}, {len(height_of_bins)} bins")
        return node_heights

    def predict(self, x : np.ndarray) -> Tuple[Number,List]:
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
            if t.feature()==0:
                walk(t.left,  (bbox[0],bbox[1],s,bbox[3]))
                walk(t.right, (s,bbox[1],bbox[2],bbox[3]))
            else:
                walk(t.left,  (bbox[0],bbox[1],bbox[2],s))
                walk(t.right, (bbox[0],s,bbox[2],bbox[3]))

        # create bounding box in feature space (not zeroed)
        f1_values = self.X_train[:, 0]
        f2_values = self.X_train[:, 1]
        overall_bbox = (np.min(f1_values), np.min(f2_values), # x,y of lower left edge
                        np.max(f1_values), np.max(f2_values)) # x,y of upper right edge
        walk(self.root, overall_bbox)

        return bboxes

    @staticmethod
    def node_samples(tree_model, data) -> Mapping[int, list]:
        """
        Return dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        # Doc say: "Return a node indicator matrix where non zero elements
        #           indicates that the samples goes through the nodes."
        dec_paths = tree_model.decision_path(data)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            _, nz_nodes = dec.nonzero()
            for node_id in nz_nodes:
                node_to_samples[node_id].append(sample_i)

        return node_to_samples

    @staticmethod
    def get_node_type(_tree_model):
        """Determine the nodes type from the tree.

        The array node_type[i]=True if node with index=i is a leaf,
        or False if the node is a split node.
        """

        node_type = np.zeros(shape=_tree_model.tree_.node_count, dtype=bool)
        stack = [(0)]  # the root node id
        while len(stack) > 0:
            node_id = stack.pop()
            # If we have a split node
            if _tree_model.tree_.children_left[node_id] != _tree_model.tree_.children_right[node_id]:
                stack.append((_tree_model.tree_.children_left[node_id]))
                stack.append((_tree_model.tree_.children_right[node_id]))
            else:
                node_type[node_id] = True  # we have a leaf node

        return node_type

    @staticmethod
    def get_leaf_sample_counts(_tree_model, min_samples=0, max_samples=None):
        """Get the number of samples for each leaf.

        There is the option to filter the leaves with less than min_samples or more than max_samples.

        :param min_samples: int
            Min number of samples for a leaf
        :param max_samples: int
            Max number of samples for a leaf

        :return: tuple
            Contains a numpy array of leaf ids and an array of leaf samples
        """

        node_type = ShadowDecTree.get_node_type(_tree_model)
        n_node_samples = _tree_model.tree_.n_node_samples

        max_samples = max_samples if max_samples else n_node_samples.max()
        leaf_samples = [(i, n_node_samples[i]) for i in range(0, _tree_model.tree_.node_count) if node_type[i]
                        and min_samples <= n_node_samples[i] <= max_samples]
        x, y = zip(*leaf_samples)
        return np.array(x), np.array(y)

    @staticmethod
    def get_leaf_sample_counts_by_class(_tree_model):
        """Get the number of samples by class for each leaf.

        :return: tuple
            Contains a list of leaf ids and a two lists of leaf samples (one for each class)
        """

        node_type = ShadowDecTree.get_node_type(_tree_model)
        sample_values = _tree_model.tree_.value

        leaf_samples = [(i, sample_values[i][0][0], sample_values[i][0][1])
                        for i in range(0, _tree_model.tree_.node_count) if (node_type[i])]

        index, leaf_samples_0, leaf_samples_1 = zip(*leaf_samples)
        return index, leaf_samples_0, leaf_samples_1

    def __str__(self):
        return str(self.root)


class ShadowDecTreeNode:
    """
    A node in a shadow tree.  Each node has left and right
    pointers to child nodes, if any.  As part of tree construction process, the
    samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """
    def __init__(self, shadow_tree, id, left=None, right=None):
        self.shadow_tree = shadow_tree
        self.id = id
        self.left = left
        self.right = right

    def split(self) -> (int,float):
        return self.shadow_tree.tree_model.tree_.threshold[self.id]

    def feature(self) -> int:
        return self.shadow_tree.tree_model.tree_.feature[self.id]

    def feature_name(self) -> (str,None):
        if self.shadow_tree.feature_names is not None:
            return self.shadow_tree.feature_names[ self.feature()]
        return None

    def samples(self) -> List[int]:
        """
        Return a list of sample indexes associated with this node. If this is a
        leaf node, it indicates the samples used to compute the predicted value
        or class.  If this is an internal node, it is the number of samples used
        to compute the split point.
        """
        return self.shadow_tree.node_to_samples[self.id]

    def nsamples(self) -> int:
        """
        Return the number of samples associated with this node. If this is a
        leaf node, it indicates the samples used to compute the predicted value
        or class. If this is an internal node, it is the number of samples used
        to compute the split point.
        """
        return self.shadow_tree.tree_model.tree_.n_node_samples[self.id] # same as len(self.node_samples)

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

    def isclassifier(self):
        return self.shadow_tree.tree_model.tree_.n_classes > 1

    def prediction(self) -> (Number,None):
        """
        If this is a leaf node, return the predicted continuous value, if this is a
        regressor, or the class number, if this is a classifier.
        """
        if not self.isleaf(): return None
        if self.isclassifier():
            counts = np.array(self.shadow_tree.tree_model.tree_.value[self.id][0])
            predicted_class = np.argmax(counts)
            return predicted_class
        else:
            return self.shadow_tree.tree_model.tree_.value[self.id][0][0]

    def prediction_name(self) -> (str,None):
        """
        If the tree model is a classifier and we know the class names,
        return the class name associated with the prediction for this leaf node.
        Return prediction class or value otherwise.
        """
        if self.isclassifier():
            if self.shadow_tree.class_names is not None:
                return self.shadow_tree.class_names[self.prediction()]
        return self.prediction()

    def class_counts(self) -> (List[int],None):
        """
        If this tree model is a classifier, return a list with the count
        associated with each class.
        """
        if self.isclassifier():
            if self.shadow_tree.class_weight is None:
                return np.array(np.round(self.shadow_tree.tree_model.tree_.value[self.id][0]), dtype=int)
            else:
                return np.round(self.shadow_tree.tree_model.tree_.value[self.id][0]/self.shadow_tree.class_weights).astype(int)
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "<pred={value},n={n}>".format(value=round(self.prediction(),1), n=self.nsamples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(),1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')
