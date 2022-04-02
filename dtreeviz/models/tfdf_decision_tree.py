from collections import defaultdict
from typing import List, Mapping

import numpy as np
from tensorflow_decision_forests.component.py_tree.node import LeafNode
from tensorflow_decision_forests.keras import RandomForestModel
from tensorflow_decision_forests.tensorflow.core import Task

from dtreeviz.models.shadow_decision_tree import ShadowDecTree, VisualisationNotYetSupportedError
from tensorflow_decision_forests.component.inspector.inspector import _RandomForestInspector


class ShadowTFDFTree(ShadowDecTree):
    NO_FEATURE = -2

    # TODO check for the other types of ensamble trees
    def __init__(self, model: RandomForestModel,
                 tree_index: int,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None
                 ):
        self.model = model
        if not self.is_fit():
            raise Exception("Model is not fit yet !")

        self.tree = self.model.make_inspector().extract_tree(tree_idx=tree_index)
        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info()
        self.features = None  # lazy initialization

        super().__init__(model, x_data, y_data, feature_names, target_name, class_names)

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
        raise VisualisationNotYetSupportedError("get_class_weights()", "TensorFlow Decision Forests")

    def get_thresholds(self) -> np.ndarray:
        raise VisualisationNotYetSupportedError("get_thresholds()", "TensorFlow Decision Forests")

    def get_features(self) -> np.ndarray:
        if self.features is not None:
            return self.features

        feature_index = [self.__class__.NO_FEATURE] * len(self.tree_nodes)
        for index, node in self.tree_nodes.items():
            if hasattr(node, "condition"):
                feature_name = node.condition._feature.name
                feature_index[index] = self.feature_names.index(feature_name)

        self.features = np.array(feature_index)
        return self.features

    def criterion(self) -> str:
        raise VisualisationNotYetSupportedError("criterion()", "TensorFlow Decision Forests")

    def get_class_weight(self):
        raise VisualisationNotYetSupportedError("get_class_weight()", "TensorFlow Decision Forests")

    def nclasses(self) -> int:
        if not self.is_classifier():
            return 1
        else:
            # didn't find an API method from TF-DF to return the class labels
            return len(np.unique(self.y_data))

    def classes(self) -> np.ndarray:
        if self.is_classifier():
            return np.unique(self.y_data)

    def get_node_samples(self):
        raise VisualisationNotYetSupportedError("get_node_samples()", "TensorFlow Decision Forests")

    def get_split_samples(self, id):
        raise VisualisationNotYetSupportedError("get_split_samples()", "TensorFlow Decision Forests")

    def get_node_nsamples(self, id):
        raise VisualisationNotYetSupportedError("get_node_nsamples()", "TensorFlow Decision Forests")

    def get_node_split(self, id) -> (int, float):
        raise VisualisationNotYetSupportedError("get_node_split()", "TensorFlow Decision Forests")

    def get_node_feature(self, id) -> int:
        return self.get_features()[id]

    def get_node_nsamples_by_class(self, id):
        raise VisualisationNotYetSupportedError("get_node_nsamples_by_class()", "TensorFlow Decision Forests")

    def get_prediction(self, id):
        raise VisualisationNotYetSupportedError("get_prediction()", "TensorFlow Decision Forests")

    def nnodes(self) -> int:
        raise VisualisationNotYetSupportedError("nnodes()", "TensorFlow Decision Forests")

    def get_node_criterion(self, id):
        raise VisualisationNotYetSupportedError("get_node_criterion()", "TensorFlow Decision Forests")

    def get_feature_path_importance(self, node_list):
        raise VisualisationNotYetSupportedError("get_feature_path_importance()", "TensorFlow Decision Forests")

    def get_max_depth(self) -> int:
        raise VisualisationNotYetSupportedError("get_max_depth()", "TensorFlow Decision Forests")

    def get_score(self) -> float:
        raise VisualisationNotYetSupportedError("get_score()", "TensorFlow Decision Forests")

    def get_min_samples_leaf(self) -> (int, float):
        raise VisualisationNotYetSupportedError("get_min_samples_leaf()", "TensorFlow Decision Forests")

    def shouldGoLeftAtSplit(self, id, x):
        raise VisualisationNotYetSupportedError("shouldGoLeftAtSplit()", "TensorFlow Decision Forests")

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




