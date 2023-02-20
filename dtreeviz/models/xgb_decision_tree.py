import json
import math
from collections import defaultdict
from typing import List, Mapping

import numpy as np

from dtreeviz.models.shadow_decision_tree import VisualisationNotYetSupportedError
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz import utils

import xgboost as xgb
from xgboost.core import Booster


class ShadowXGBDTree(ShadowDecTree):
    LEFT_CHILDREN_COLUMN = "Yes"
    RIGHT_CHILDREN_COLUMN = "No"
    NODE_COLUMN = "Node"
    NO_CHILDREN = -1
    NO_SPLIT = -2
    NO_FEATURE = -2
    ROOT_NODE = 0

    def __init__(self, booster: Booster,
                 tree_index: int,
                 X_train,
                 y_train,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None
                 ):
        if hasattr(booster, 'get_booster'):
            booster = booster.get_booster() # support XGBClassifier and XGBRegressor
        utils.check_tree_index(tree_index, len(booster.get_dump()))
        self.booster = booster
        self.tree_index = tree_index
        self.tree_to_dataframe = self._get_tree_dataframe()
        self.children_left = self._calculate_children(self.__class__.LEFT_CHILDREN_COLUMN)
        self.children_right = self._calculate_children(self.__class__.RIGHT_CHILDREN_COLUMN)
        self.config = json.loads(self.booster.save_config())
        self.node_to_samples = None  # lazy initialized
        self.features = None  # lazy initialized

        super().__init__(booster, X_train, y_train, feature_names, target_name, class_names)

    def is_fit(self):
        return isinstance(self.booster, Booster)

    # TODO - add implementation
    def get_class_weights(self):
        return None

    # TODO - add implementation
    def get_class_weight(self):
        return None

    def criterion(self):
        raise VisualisationNotYetSupportedError("criterion()", "XGBoost")

    def get_children_left(self):
        return self._calculate_children(self.__class__.LEFT_CHILDREN_COLUMN)

    def get_children_right(self):
        return self._calculate_children(self.__class__.RIGHT_CHILDREN_COLUMN)

    def get_node_split(self, id) -> (float):
        """
        Split values could not be the same like in plot_tree(booster). This is because xgb_model_classifier.joblib.trees_to_dataframe()
        get data using dump_format = text from xgb_model_classifier.joblib.get_dump()
        """
        node_split = self._get_nodes_values("Split")[id]
        return node_split if not math.isnan(node_split) else self.__class__.NO_SPLIT

    def get_node_feature(self, id) -> int:
        feature_name = self._get_nodes_values("Feature")[id]
        try:
            return self.feature_names.index(feature_name)
        except ValueError as error:
            return self.__class__.NO_FEATURE

    def get_features(self):
        if self.features is not None:
            return self.features

        nodes = self._get_column_value(self.NODE_COLUMN)
        self.features = {node: self.get_node_feature(node) for node in nodes}
        return self.features

    def get_node_samples(self):
        """
        Return dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """

        if self.node_to_samples is not None:
            return self.node_to_samples

        prediction_leaves = self.booster.predict(xgb.DMatrix(self.X_train, feature_names=self.feature_names),
                                                 pred_leaf=True)

        if len(prediction_leaves.shape) > 1:
            prediction_leaves = prediction_leaves[:, self.tree_index]

        node_to_samples = defaultdict(list)
        for sample_i, prediction_leaf in enumerate(prediction_leaves):
            prediction_path = self._get_leaf_prediction_path(prediction_leaf)
            for node_id in prediction_path:
                node_to_samples[node_id].append(sample_i)

        self.node_to_samples = node_to_samples
        return node_to_samples

    def get_split_samples(self, id):
        samples = np.array(self.get_node_samples()[id])
        node_X_data = self.X_train[samples, self.get_node_feature(id)]
        split = self.get_node_split(id)

        left = np.nonzero(node_X_data < split)[0]
        right = np.nonzero(node_X_data >= split)[0]

        return left, right

    def get_root_edge_labels(self):
        return ["&lt;", "&ge;"]

    def get_node_nsamples(self, id):
        return len(self.get_node_samples()[id])

    def _get_leaf_prediction_path(self, leaf):
        prediction_path = [leaf]
        left_parent = np.array(list(self.children_left.keys()))
        left_child = np.array(list(self.children_left.values()))

        right_parent = np.array(list(self.children_right.keys()))
        right_child = np.array(list(self.children_right.values()))

        def walk(node_id):
            if node_id != self.__class__.ROOT_NODE:
                try:
                    parent_node = left_parent[np.where(left_child == node_id)[0][0]]
                    prediction_path.append(parent_node)
                    walk(parent_node)
                except IndexError:
                    pass

                try:
                    parent_node = right_parent[np.where(right_child == node_id)[0][0]]
                    prediction_path.append(parent_node)
                    walk(parent_node)
                except IndexError:
                    pass

        walk(leaf)
        return prediction_path

    def _get_tree_dataframe(self):
        return self.booster.trees_to_dataframe().query(f"Tree == {self.tree_index}")

    def _get_column_value(self, column_name):
        return self.tree_to_dataframe[column_name].to_numpy()

    def _get_nodes_values(self, column_name):
        nodes = self._get_column_value(self.NODE_COLUMN)
        nodes_values = self._get_column_value(column_name)

        return dict(zip(nodes, nodes_values))

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
        nodes = self._get_column_value(self.NODE_COLUMN)
        return dict(zip(nodes, map(int, children)))

    def get_feature_path_importance(self, node_list):
        raise VisualisationNotYetSupportedError("get_feature_path_importance()", "XGBoost")

    def get_node_criterion(self):
        raise VisualisationNotYetSupportedError("get_node_criterion()", "XGBoost")

    # TODO check explain method for both classification and not classification splits
    def get_thresholds(self):
        nodes = self._get_column_value(self.NODE_COLUMN)
        thresholds = {node: self.get_node_split(node) for node in nodes}
        return thresholds

    def get_node_nsamples_by_class(self, id):
        all_nodes = self.internal + self.leaves
        if self.is_classifier():
            node_value = [node.n_sample_classes() for node in all_nodes if node.id == id]
            return node_value[0]

    def get_prediction(self, id):
        all_nodes = self.internal + self.leaves
        if self.is_classifier():
            node_value = [node.n_sample_classes() for node in all_nodes if node.id == id]
            return np.argmax(node_value[0])
        elif not self.is_classifier():
            node_samples = [node.samples() for node in all_nodes if node.id == id][0]
            return np.mean(self.y_train[node_samples])

    def is_classifier(self):
        objective_name = self.config["learner"]["objective"]["name"].split(":")[0]
        if objective_name == "binary" or objective_name == "multi":
            return True
        elif objective_name == "reg":
            return False
        return None

    def nnodes(self):
        return self.tree_to_dataframe.shape[0]

    def nclasses(self):
        if not self.is_classifier():
            return 1
        else:
            return len(np.unique(self.y_train))

    def classes(self):
        if self.is_classifier():
            return np.unique(self.y_train)

    def get_max_depth(self):
        return int(self.config["learner"]["gradient_booster"]["updater"]["prune"]["train_param"]["max_depth"])

    def get_score(self):
        raise VisualisationNotYetSupportedError("get_score()", "XGBoost")

    def get_min_samples_leaf(self):
        raise VisualisationNotYetSupportedError("get_min_samples_leaf()", "XGBoost")

    def shouldGoLeftAtSplit(self, id, x):
        return x < self.get_node_split(id)
