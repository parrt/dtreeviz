from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from collections import defaultdict
import numpy as np
from sklearn.utils import compute_class_weight



class DTree:
    def __init__(self):
        pass


class SKDTree(DTree):
    def __init__(self, tree_model):
        self.tree_model: (DecisionTreeClassifier, DecisionTreeRegressor) = tree_model

    def is_fit(self):
        return getattr(self.tree_model, 'tree_') is not None

    # TODO
    # check results with original shadow.py
    def get_class_weight(self, y_train):
        if self.tree_model.tree_.n_classes > 1:
            unique_target_values = np.unique(y_train)
            return compute_class_weight(self.tree_model.class_weight, unique_target_values, y_train)
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

class XGBDTree(DTree):
    def __init__(self):
        pass
