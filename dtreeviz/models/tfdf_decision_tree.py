from collections import defaultdict
from typing import List, Mapping

from tensorflow_decision_forests.component.py_tree.node import LeafNode
from tensorflow_decision_forests.keras import RandomForestModel

from dtreeviz.models.shadow_decision_tree import ShadowDecTree


class ShadowTFDFTree:

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

        self.tree = model.make_inspector().extract_tree(tree_idx=tree_index)

        self.tree_nodes, self.children_left, self.children_right = self._get_nodes_info(self.tree)


        # super(ShadowTFDFTree, self).__init__(model, x_data, y_data, feature_names, target_name, class_names)


    def is_fit(self) -> bool:
        try:
            self.model.make_inspector()
            return True
        except Exception:
            return False

    def get_children_left(self):
        pass

    def get_children_right(self):
        pass

    def _get_nodes_info(self, model):
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

        recur(model.root, 0)

        return tree_nodes, children_left, children_right

