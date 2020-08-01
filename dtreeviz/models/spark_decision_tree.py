from abc import ABC
from typing import List, Mapping

import numpy as np

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from pyspark.ml.classification import DecisionTreeClassificationModel, DecisionTreeRegressionModel


class ShadowSparkTree(ShadowDecTree):

    def __init__(self, tree_model: (DecisionTreeClassificationModel, DecisionTreeRegressionModel),
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):

        self.tree_model = tree_model
        self.tree_nodes, self.children_left, self.children_right = self._get_node_info()
        super().__init__(tree_model, x_data, y_data)

        pass

    def _get_node_info(self):
        tree_nodes = [None] * self.tree_model.numNodes
        children_left = [-1] * self.tree_model.numNodes
        children_right = [-1] * self.tree_model.numNodes
        node_index = 0

        def recur(node, node_id):
            nonlocal node_index
            tree_nodes[node_id] = node
            if node.numDescendants() == 0:
                return
            else:
                node_index += 1
                children_left[node_id] = node_index
                # print(f"node, node_left {node_id, node_index}")
                recur(node.leftChild(), node_index)

                node_index += 1
                children_right[node_id] = node_index
                # print(f"node, node_right {node_id, node_index}")
                recur(node.rightChild(), node_index)

        recur(self.tree_model._call_java('rootNode'), 0)
        return tree_nodes, children_left, children_right

    def is_fit(self) -> bool:
        if isinstance(self.tree_model, (DecisionTreeClassificationModel, DecisionTreeRegressionModel)):
            return True
        return False

    def is_classifier(self) -> bool:
        pass

    def get_class_weights(self):
        pass

    def get_thresholds(self) -> np.ndarray:
        pass

    def get_features(self) -> np.ndarray:
        pass

    def criterion(self) -> str:
        return self.tree_model.getImpurity().upper()

    def get_class_weight(self):
        pass

    def nclasses(self) -> int:
        pass

    def classes(self) -> np.ndarray:
        pass

    def get_node_samples(self):
        pass

    def get_children_left(self) -> np.ndarray:
        return np.array(self.children_left, dtype=int)

    def get_children_right(self):
        return np.array(self.children_right, dtype=int)

    def get_node_split(self, id) -> (int, float):
        pass

    def get_node_feature(self, id) -> int:
        pass

    def get_prediction_value(self, id):
        pass

    def nnodes(self) -> int:
        pass

    def get_node_criterion(self, id):
        return self.tree_nodes[id].impurity()

    def get_feature_path_importance(self, node_list):
        pass

    def get_max_depth(self) -> int:
        pass

    def get_score(self) -> float:
        pass

    def get_min_samples_leaf(self) -> (int, float):
        pass
