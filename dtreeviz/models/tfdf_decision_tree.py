from collections import defaultdict
from typing import List, Mapping

import numpy as np
from tensorflow_decision_forests.component.py_tree.node import LeafNode
from tensorflow_decision_forests.keras import RandomForestModel
from tensorflow_decision_forests.tensorflow.core import Task

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from tensorflow_decision_forests.component.inspector.inspector import _RandomForestInspector


class ShadowTFDFTree(ShadowDecTree):

    # TODO check for the other types of ensamble trees
    def __init__(self, model: RandomForestModel,
                 tree_index: int,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None
                 ):

        # TODO read about init and pytest

        self.model = model
        if not self.is_fit():
            raise Exception("Model is not fit yet !")

        self.tree = self.model.make_inspector().extract_tree(tree_idx=tree_index)

        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info()

        super(ShadowTFDFTree, self).__init__(model, x_data, y_data, feature_names, target_name, class_names)

    def is_fit(self) -> bool:
        try:
            self.model.make_inspector()
            return True
        except Exception:
            return False

    def get_children_left(self):
        return self.children_left

    def get_children_right(self):
        return self.children_right

    def is_classifier(self) -> bool:
        return self.model._task == Task.CLASSIFICATION


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

    def get_split_samples(self, id):
        pass

    def get_node_nsamples(self, id):
        pass

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

    def _get_nodes_info(self):
        """
        Get individual node info and left/right child node. We are using a dict as a data structure to keep
        the left and right child node info.
        """

        tree_nodes = defaultdict(lambda: None)
        children_left = defaultdict(lambda: -1)
        children_right = defaultdict(lambda: -1)
        node_index = 0

        def recur(node, node_id):
            nonlocal node_index

            tree_nodes[node_id] = node

            if isinstance(node, LeafNode):
                return
            else:
                node_index += 1
                children_left[node_id] = node_index
                recur(node.neg_child, node_index)

                node_index += 1
                children_right[node_id] = node_index
                recur(node.pos_child, node_index)

        recur(self.tree.root, 0)

        return tree_nodes, children_left, children_right




