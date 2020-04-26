from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import compute_class_weight
from xgboost.core import Booster
import pandas as pd
import math

class SKDTree:
    def __init__(self, tree_model):
        self.tree_model: (DecisionTreeClassifier, DecisionTreeRegressor) = tree_model

    def is_fit(self):
        return getattr(self.tree_model, 'tree_') is not None

    # TODO
    # check results with original shadow.py
    def get_class_weights(self, y_train):
        if self.tree_model.tree_.n_classes > 1:
            unique_target_values = np.unique(y_train)
            return compute_class_weight(self.tree_model.class_weight, unique_target_values, y_train)
        return self.tree_model.class_weight

    def get_class_weight(self):
        return self.tree_model.class_weight

    def get_n_classes(self):
        return self.tree_model.tree_.n_classes[0]

    def get_node_samples(self, data):
        """
        Return dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        # Doc say: "Return a node indicator matrix where non zero elements
        #           indicates that the samples goes through the nodes."

        dec_paths = self.tree_model.decision_path(data)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            _, nz_nodes = dec.nonzero()
            for node_id in nz_nodes:
                node_to_samples[node_id].append(sample_i)

        return node_to_samples

    def get_children_left(self):
        return self.tree_model.tree_.children_left

    def get_children_right(self):
        return self.tree_model.tree_.children_right

    def get_node_split(self, id) -> (int, float):
        return self.tree_model.tree_.threshold[id]

    def get_node_feature(self, id) -> int:
        return self.tree_model.tree_.feature[id]

    def get_value(self, id):
        return self.tree_model.tree_.value[id][0]

    def get_node_count(self):
        return self.tree_model.tree_.node_count

    # def get_leaf_sample_counts(self, leaves):
    #     return [self.tree_model.tree_.n_node_samples[leaf.id] for leaf in leaves]

    def get_node_nsamples(self, id) -> int:
        return self.tree_model.tree_.n_node_samples[id]

    def get_node_criterion(self, id):
        return self.tree_model.tree_.impurity[id]

    def get_prediction_value(self, id):
        return self.tree_model.tree_.value[id]


class XGBDTree:
    LEFT_CHILDREN_COLUMN = "Yes"
    RIGHT_CHILDREN_COLUMN = "No"
    NO_CHILDREN = -1
    NO_SPLIT = -2
    NO_FEATURE = -2

    def __init__(self, booster: Booster,
                 tree_index: int,
                 data: pd.DataFrame = None):
        self.booster = booster
        self.tree_index = tree_index
        self.data = data
        self.tree_to_dataframe = self._get_tree_dataframe()
        self.children_left = self._calculate_children(self.__class__.LEFT_CHILDREN_COLUMN)
        self.children_right = self._calculate_children(self.__class__.RIGHT_CHILDREN_COLUMN)

    def is_fit(self):
        return isinstance(self.booster, Booster)

    def get_class_weight(self):
        return None

    def get_children_left(self):
        return self.children_left

    def get_children_right(self):
        return self.children_right

    def get_node_split(self, id) -> (float):
        node_split = self._get_column_value("Split")[id]
        return node_split if not math.isnan(node_split) else self.__class__.NO_SPLIT

    def get_node_feature(self, id) -> int:
        feature_name = self._get_column_value("Feature")[id]
        try:
            return self.booster.feature_names.index(feature_name)
        except ValueError as error:
            return self.__class__.NO_FEATURE

    def get_node_samples(self, data):
        """
        Return dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        # Doc say: "Return a node indicator matrix where non zero elements
        #           indicates that the samples goes through the nodes."

        dec_paths = self.tree_model.decision_path(data)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            _, nz_nodes = dec.nonzero()
            for node_id in nz_nodes:
                node_to_samples[node_id].append(sample_i)

        return node_to_samples

    def _get_tree_dataframe(self):
        return self.booster.trees_to_dataframe().query(f"Tree == {self.tree_index}")

    def _get_column_value(self, column_name):
        return self.tree_to_dataframe[column_name].to_numpy()

    def _split_column_value(self, column_name):
        def split_value(value):
            if isinstance(value, str):
                return value.split("-")[1]
            else:
                return value

        return self.tree_to_dataframe.apply(lambda row: split_value(row.get(f"{column_name}")), axis=1)

    def _change_no_children_value(self, children):
        return children.fillna(self.__class__.NO_CHILDREN)

    def _calculate_children(self, column_name):
        children = self._split_column_value(column_name)
        children = self._change_no_children_value(children)
        return children.to_numpy(dtype=int)
