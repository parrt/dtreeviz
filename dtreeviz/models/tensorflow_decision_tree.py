from collections import defaultdict
from typing import List, Mapping

import numpy as np
import tensorflow_decision_forests
from tensorflow_decision_forests.component.py_tree.node import LeafNode
from tensorflow_decision_forests.keras import RandomForestModel
from tensorflow_decision_forests.tensorflow.core import Task

from dtreeviz.models.shadow_decision_tree import ShadowDecTree, VisualisationNotYetSupportedError


class ShadowTensorflowTree(ShadowDecTree):
    NO_FEATURE = -2
    NO_SPLIT = -2

    # TODO check for the other types of ensamble trees
    def __init__(self, model: RandomForestModel,
                 tree_index: int,
                 X_train,
                 y_train,
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
        self.column_dataspec = self._get_column_dataspec()
        self.node_to_samples = None  # lazy initialization
        self.thresholds = None  # lazy  initialization

        super().__init__(model, X_train, y_train, feature_names, target_name, class_names)

    def _get_column_dataspec(self):
        column_dataspec = {}
        for column_spec in self.model.make_inspector().dataspec.columns:
            column_dataspec[column_spec.name] = column_spec
        return column_dataspec

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
        if tensorflow_decision_forests.__version__<'1.2.0':
            return self.model._task == Task.CLASSIFICATION
        return self.model.task == Task.CLASSIFICATION

    def get_class_weights(self):
        raise VisualisationNotYetSupportedError("get_class_weights()", "TensorFlow Decision Forests")

    def get_thresholds(self) -> np.ndarray:
        if self.thresholds is not None:
            return self.thresholds

        thresholds = [self.__class__.NO_SPLIT] * len(self.tree_nodes)
        for index, node in self.tree_nodes.items():
            if hasattr(node, "condition"):
                node_condition = node.condition
                if hasattr(node_condition, "threshold"):
                    thresholds[index] = node.condition.threshold
                #     for conditional split
                # this threshold contains the right condition path
                if hasattr(node_condition, "mask"):
                    thresholds[index] = node.condition.mask
                    feature_split_name = node.condition.feature.name
                    if self.column_dataspec[feature_split_name].categorical.offset_value_by_one_during_training is True:
                        thresholds[index] = [value - 1 for value in thresholds[index]]

        self.thresholds = np.array(thresholds, dtype=object)
        return self.thresholds

    def get_features(self) -> np.ndarray:
        if self.features is not None:
            return self.features

        feature_index = [self.__class__.NO_FEATURE] * len(self.tree_nodes)
        for index, node in self.tree_nodes.items():
            if hasattr(node, "condition"):
                feature_name = node.condition.feature.name
                feature_index[index] = self.feature_names.index(feature_name)

        self.features = np.array(feature_index)
        return self.features

    def criterion(self) -> str:
        raise VisualisationNotYetSupportedError("criterion()", "TensorFlow Decision Forests")

    # TODO check if we need to implement it
    def get_class_weight(self):
        return None

    def nclasses(self) -> int:
        if not self.is_classifier():
            return 1
        else:
            # didn't find an API method from TF-DF to return the class labels
            return len(np.unique(self.y_train))

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
            left = np.nonzero(indices == 0)[0]
            right = np.nonzero(indices == 1)[0]
        else:
            left = np.nonzero(node_X_data < split)[0]
            right = np.nonzero(node_X_data >= split)[0]
        return left, right

    def get_node_nsamples(self, id):
        return len(self.get_node_samples()[id])

    def get_node_split(self, id) -> (int, float):
        return self.get_thresholds()[id]

    def get_node_feature(self, id) -> int:
        return self.get_features()[id]

    # TODO check if we can put this method in the super class
    def get_node_nsamples_by_class(self, id):
        all_nodes = self.internal + self.leaves
        if self.is_classifier():
            node_value = [node.n_sample_classes() for node in all_nodes if node.id == id]
            return node_value[0]

    def get_prediction(self, id):
        if self.is_classifier():
            # In a GBT model, the trees are always regressive trees (even if the GBT is a classifier). So we don't
            # have the probability attribute
            if "tensorflow_decision_forests.keras.GradientBoostedTreesModel" in str(type(self.model)):
                return self.tree_nodes[id].value.value
            return np.argmax(self.tree_nodes[id].value.probability)
        else:
            return self.tree_nodes[id].value.value

    def is_categorical_split(self, id) -> bool:
        node_condition = self.tree_nodes[id].condition

        if hasattr(node_condition, "threshold"):
            return False
        return True

    def nnodes(self) -> int:
        raise VisualisationNotYetSupportedError("nnodes()", "TensorFlow Decision Forests")

    def get_node_criterion(self, id):
        raise VisualisationNotYetSupportedError("get_node_criterion()", "TensorFlow Decision Forests")

    def get_feature_path_importance(self, node_list):
        raise VisualisationNotYetSupportedError("get_feature_path_importance()", "TensorFlow Decision Forests")

    def get_max_depth(self) -> int:
        return self.model._learner_params["max_depth"]

    def get_score(self) -> float:
        raise VisualisationNotYetSupportedError("get_score()", "TensorFlow Decision Forests")

    def get_min_samples_leaf(self) -> (int, float):
        raise VisualisationNotYetSupportedError("get_min_samples_leaf()", "TensorFlow Decision Forests")

    def shouldGoLeftAtSplit(self, id, x):
        if self.is_categorical_split(id):
            return x not in self.get_node_split(id)
        return x < self.get_node_split(id)

    def get_root_edge_labels(self):
        return ["&lt;", "&ge;"]

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
