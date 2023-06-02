from collections import defaultdict
from typing import List, Mapping

import numpy as np
from sklearn.utils import compute_class_weight

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.utils import criterion_remapping


class ShadowSKDTree(ShadowDecTree):
    def __init__(self, tree_model,
                 X_train,
                 y_train,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):

        self.node_to_samples = None
        super().__init__(tree_model, X_train, y_train, feature_names, target_name, class_names)

    def is_fit(self):
        return getattr(self.tree_model, 'tree_') is not None

    def is_classifier(self):
        return self.nclasses() > 1

    def get_class_weights(self):
        if self.is_classifier():
            unique_target_values = np.unique(self.y_train)
            return compute_class_weight(self.tree_model.class_weight, classes=unique_target_values, y=self.y_train)

    def get_thresholds(self):
        return self.tree_model.tree_.threshold

    def get_features(self):
        return self.tree_model.tree_.feature

    def criterion(self):
        return criterion_remapping(self.tree_model.criterion)

    def get_class_weight(self):
        return self.tree_model.class_weight

    def nclasses(self):
        return self.tree_model.tree_.n_classes[0]

    def classes(self):
        if self.is_classifier():
            return self.tree_model.classes_

    def get_node_samples(self):
        if self.node_to_samples is not None:
            return self.node_to_samples

        dec_paths = self.tree_model.decision_path(self.X_train)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            _, nz_nodes = dec.nonzero()
            for node_id in nz_nodes:
                node_to_samples[node_id].append(sample_i)

        self.node_to_samples = node_to_samples
        return node_to_samples

    def get_split_samples(self, id):
        samples = np.array(self.get_node_samples()[id])
        node_X_data = self.X_train[samples, self.get_node_feature(id)]
        split = self.get_node_split(id)

        left = np.nonzero(node_X_data <= split)[0]
        right = np.nonzero(node_X_data > split)[0]

        return left, right

    def get_root_edge_labels(self):
        return ["&le;", "&gt;"]

    def get_node_nsamples(self, id):
        return len(self.get_node_samples()[id])

    def get_children_left(self):
        return self.tree_model.tree_.children_left

    def get_children_right(self):
        return self.tree_model.tree_.children_right

    def get_node_split(self, id) -> (int, float):
        return self.tree_model.tree_.threshold[id]

    def get_node_feature(self, id) -> int:
        return self.tree_model.tree_.feature[id]

    def get_node_nsamples_by_class(self, id):
        # This is the code to return the nsamples/class from tree metadata. It's faster, but the visualisations cannot
        # be made on new datasets.
        # if self.is_classifier():
        #     return self.tree_model.tree_.value[id][0]

        # This code allows us to return the nsamples/class based on a dataset, train or validation
        if self.is_classifier():
            all_nodes = self.internal + self.leaves
            node_value = [node.n_sample_classes() for node in all_nodes if node.id == id]
            if self.get_class_weights() is None:
                return node_value[0]
            else:
                return node_value[0] * self.get_class_weights()

    def get_prediction(self, id):
        if self.is_classifier():
            counts = self.tree_model.tree_.value[id][0]
            return np.argmax(counts)
        else:
            return self.tree_model.tree_.value[id][0][0]

    def nnodes(self):
        return self.tree_model.tree_.node_count

    def get_node_criterion(self, id):
        return self.tree_model.tree_.impurity[id]

    def get_feature_path_importance(self, node_list):
        gini = np.zeros(self.tree_model.tree_.n_features)
        tree_ = self.tree_model.tree_
        for node in node_list:
            if self.tree_model.tree_.children_left[node] != -1:
                node_left = self.tree_model.tree_.children_left[node]
                node_right = self.tree_model.tree_.children_right[node]
                gini[tree_.feature[node]] += tree_.weighted_n_node_samples[node] * tree_.impurity[node] \
                                             - tree_.weighted_n_node_samples[node_left] * tree_.impurity[node_left] \
                                             - tree_.weighted_n_node_samples[node_right] * tree_.impurity[node_right]
        normalizer = np.sum(gini)
        if normalizer > 0.0:
            gini /= normalizer

        return gini

    def get_max_depth(self):
        return self.tree_model.max_depth

    def get_score(self):
        return self.tree_model.score(self.X_train, self.y_train)

    def get_min_samples_leaf(self):
        return self.tree_model.min_samples_leaf

    def shouldGoLeftAtSplit(self, id, x):
        return x <= self.get_node_split(id)
