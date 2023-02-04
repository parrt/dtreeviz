from collections import defaultdict
from typing import List, Mapping

import numpy as np
import pandas as pd
from lightgbm.basic import Booster

from dtreeviz.models.shadow_decision_tree import ShadowDecTree, VisualisationNotYetSupportedError
from dtreeviz import utils


class ShadowLightGBMTree(ShadowDecTree):

    def __init__(self,
                 booster: Booster,
                 tree_index: int,
                 X_train: (pd.DataFrame, np.ndarray),
                 y_train: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):

        utils.check_tree_index(tree_index, booster.num_trees())
        self.booster = booster
        self.tree_index = tree_index
        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info()
        self.thresholds = None  # lazy evaluation
        self.features = None  # lazy evaluation
        self.node_to_samples = None

        super().__init__(booster, X_train, y_train, feature_names, target_name, class_names)

    def _get_nodes_info(self):
        tree_nodes = {}
        children_left = {}
        children_right = {}
        node_index = 0

        def _walk_tree(node, node_id):
            nonlocal node_index
            tree_nodes[node_id] = node
            if node.get("split_index") is None:
                children_left[node_id] = -1
                children_right[node_id] = -1
                return

            node_index += 1
            children_left[node_id] = node_index
            _walk_tree(node.get("left_child"), node_index)

            node_index += 1
            children_right[node_id] = node_index
            _walk_tree(node.get("right_child"), node_index)

        def _convert_dict_to_list(my_dict):
            my_list = [-1] * len(my_dict)
            for key, value in my_dict.items():
                my_list[key] = value
            return my_list

        tree_dump = self.booster.dump_model()["tree_info"][self.tree_index]
        _walk_tree(tree_dump["tree_structure"], node_index)

        for node in tree_nodes.values():
            node.pop("left_child", None)
            node.pop("right_child", None)

        children_left_list = _convert_dict_to_list(children_left)
        children_right_list = _convert_dict_to_list(children_right)
        tree_node_list = _convert_dict_to_list(tree_nodes)

        return tree_node_list, children_left_list, children_right_list

    def is_fit(self) -> bool:
        return isinstance(self.booster, Booster)

    def is_classifier(self) -> bool:
        objective = self.booster.dump_model(num_iteration=1)["objective"]
        if "binary" in objective or "multiclass" in objective:
            return True
        elif objective in ["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma",
                           "tweedie"]:
            return False
        raise Exception(f"objective {objective} is not yet supported by dtreeviz's lightgbm implementation")

    def is_categorical_split(self, id) -> bool:
        node = self.tree_nodes[id]
        if 'split_index' in node:
            if node["decision_type"] == "==":
                return True
        return False

    def get_class_weights(self):
        pass

    def get_thresholds(self) -> np.ndarray:
        if self.thresholds is not None:
            return self.thresholds

        node_thresholds = [-1] * self.nnodes()
        for i in range(self.nnodes()):
            if self.children_left[i] != -1 and self.children_right[i] != -1:
                if self.is_categorical_split(i):
                    node_thresholds[i] = list(map(int, self.tree_nodes[i]["threshold"].split("||")))
                else:
                    node_thresholds[i] = round(self.tree_nodes[i]["threshold"], 2)

        self.thresholds = np.array(node_thresholds, dtype=object)
        return self.thresholds

    def get_features(self) -> np.ndarray:
        if self.features is not None:
            return self.features

        self.features = [-1] * self.nnodes()
        for i, node in enumerate(self.tree_nodes):
            self.features[i] = node.get("split_feature", -1)

        self.features = np.array(self.features)
        return self.features

    def criterion(self) -> str:
        raise VisualisationNotYetSupportedError("criterion()", "LightGBM")

    def get_class_weight(self):
        return None

    def nclasses(self) -> int:
        if self.booster._Booster__num_class == 1:
            return 2
        else:
            return self.booster._Booster__num_class

    def classes(self) -> np.ndarray:
        if self.is_classifier():
            return np.unique(self.y_train)

    def get_node_samples(self):
        if self.node_to_samples is not None:
            return self.node_to_samples

        node_to_samples = defaultdict(list)
        for i in range(self.X_train.shape[0]):
            path = self.predict_path(self.X_train[i])
            for node in path:
                node_to_samples[node.id].append(i)

        self.node_to_samples = node_to_samples
        return self.node_to_samples

    def get_split_samples(self, id):
        samples = np.array(self.get_node_samples()[id])
        node_X_data = self.X_train[samples, self.get_node_feature(id)]
        split = self.get_node_split(id)

        if self.is_categorical_split(id):
            indices = np.sum([node_X_data == split_value for split_value in self.get_node_split(id)], axis=0)
            left = np.nonzero(indices == 1)[0]
            right = np.nonzero(indices == 0)[0]
        else:
            left = np.nonzero(node_X_data <= split)[0]
            right = np.nonzero(node_X_data > split)[0]
        return left, right

    def get_root_edge_labels(self):
        return ["&le;", "&gt;"]

    def get_node_nsamples(self, id):
        if self.children_right[id] == -1 and self.children_left[id] == -1:
            return self.tree_nodes[id]["leaf_count"]
        else:
            return self.tree_nodes[id]["internal_count"]

    def get_children_left(self) -> np.ndarray:
        return np.array(self.children_left, dtype=int)

    def get_children_right(self) -> np.ndarray:
        return np.array(self.children_right, dtype=int)

    def get_node_split(self, id) -> (int, float):
        return self.get_thresholds()[id]

    def get_node_feature(self, id) -> int:
        return self.get_features()[id]

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

    def nnodes(self) -> int:
        return len(self.tree_nodes)

    def get_node_criterion(self, id):
        raise VisualisationNotYetSupportedError("get_node_criterion()", "LightGBM")

    def get_feature_path_importance(self, node_list):
        raise VisualisationNotYetSupportedError("get_feature_path_importance()", "LightGBM")

    def get_max_depth(self) -> int:
        # max_depth can be found in lgbm_model.params, but only if the max_depth is specified
        # otherwise the max depth is -1, from lgbm_model.model_to_string() (to double check)
        raise VisualisationNotYetSupportedError("get_max_depth()", "LightGBM")

    def get_score(self) -> float:
        raise VisualisationNotYetSupportedError("get_score()", "LightGBM")

    def get_min_samples_leaf(self) -> (int, float):
        default_value = 20
        return self.booster.params.get("min_data_in_leaf", default_value)

    def shouldGoLeftAtSplit(self, id, x):
        if self.is_categorical_split(id):
            return x in self.get_node_split(id)
        return x <= self.get_node_split(id)
