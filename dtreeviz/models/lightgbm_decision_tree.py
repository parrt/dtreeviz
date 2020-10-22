from typing import List, Mapping

import numpy as np
import pandas as pd
from lightgbm.basic import Booster

from dtreeviz.models.shadow_decision_tree import ShadowDecTree, VisualisationNotYetSupportedError


class ShadowLightGBMTree(ShadowDecTree):

    def __init__(self,
                 booster: Booster,
                 tree_index: int,
                 x_data: (pd.DataFrame, np.ndarray),
                 y_data: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):
        self.booster = booster
        self.tree_index = tree_index
        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info()

        super().__init__(booster, x_data, y_data, feature_names, target_name, class_names)

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
        if "binary" in objective:
            return True
        elif objective in ["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma",
                           "tweedie"]:
            return False
        raise Exception(f"objective {objective} is not yet supported by dtreeviz's lightgbm implementation")

    def get_class_weights(self):
        pass

    def get_thresholds(self) -> np.ndarray:
        pass

    def get_features(self) -> np.ndarray:
        pass

    def criterion(self) -> str:
        raise VisualisationNotYetSupportedError("criterion()", "LightGBM")

    def get_class_weight(self):
        pass

    def nclasses(self) -> int:
        pass

    def classes(self) -> np.ndarray:
        pass

    def get_node_samples(self):
        pass

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
        pass

    def get_node_feature(self, id) -> int:
        pass

    def get_node_nsamples_by_class(self, id):
        pass

    def get_prediction(self, id):
        pass

    def nnodes(self) -> int:
        pass

    def get_node_criterion(self, id):
        raise VisualisationNotYetSupportedError("get_node_criterion()", "LightGBM")

    def get_feature_path_importance(self, node_list):
        pass

    def get_max_depth(self) -> int:
        pass

    def get_score(self) -> float:
        pass

    def get_min_samples_leaf(self) -> (int, float):
        pass

    def shouldGoLeftAtSplit(self, id, x):
        pass
