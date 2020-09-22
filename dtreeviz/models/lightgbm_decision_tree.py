from typing import List, Mapping

import numpy as np
import pandas as pd
from lightgbm.basic import Booster

from dtreeviz.models.shadow_decision_tree import ShadowDecTree


class ShadowLightGBMTree(ShadowDecTree):

    def __init__(self,
                 tree_model: Booster,
                 tree_index: int,
                 x_data: (pd.DataFrame, np.ndarray),
                 y_data: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):
        self.tree_model = tree_model
        self.tree_index = tree_index
        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info()

        super().__init__(tree_model, x_data, y_data, feature_names, target_name, class_names)

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

        tree_dump = self.tree_model.dump_model()["tree_info"][self.tree_index]
        _walk_tree(tree_dump["tree_structure"], node_index)

        for node in tree_nodes.values():
            node.pop("left_child", None)
            node.pop("right_child", None)

        children_left_list = _convert_dict_to_list(children_left)
        children_right_list = _convert_dict_to_list(children_right)
        tree_node_list = _convert_dict_to_list(tree_nodes)

        return tree_node_list, children_left_list, children_right_list

    def is_fit(self) -> bool:
        return True

    def is_classifier(self) -> bool:
        pass

    def get_class_weights(self):
        pass

    def get_thresholds(self) -> np.ndarray:
        pass

    def get_features(self) -> np.ndarray:
        pass

    def criterion(self) -> str:
        pass

    def get_class_weight(self):
        pass

    def nclasses(self) -> int:
        pass

    def classes(self) -> np.ndarray:
        pass

    def get_node_samples(self):
        pass

    def get_node_nsamples(self, id):
        pass

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
        pass

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
