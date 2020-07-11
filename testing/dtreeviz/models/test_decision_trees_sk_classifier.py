import joblib
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree


@pytest.fixture()
def dec_tree() -> (DecisionTreeClassifier, DecisionTreeRegressor):
    return joblib.load("fixtures/sk_decision_tree_classifier.joblib")


@pytest.fixture()
def shadow_dec_tree(dec_tree, dataset) -> ShadowSKDTree:
    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Survived"
    class_names = list(dec_tree.classes_)
    return ShadowSKDTree(dec_tree, dataset[features], dataset[target], features, class_names)


def test_x_dataset(x_dataset_classifier):
    n_rows, n_cols = x_dataset_classifier.shape
    assert n_rows == 20, "Number of rows should be 20"
    assert n_cols == 6, "Number of columns shoud be 6"


def test_feature_number(shadow_dec_tree):
    assert shadow_dec_tree.feature_names == ['Pclass', 'Age', 'Fare', 'Sex_label', 'Cabin_label', 'Embarked_label']


def test_is_fit(shadow_dec_tree):
    assert shadow_dec_tree.is_fit() == True


def test_is_classifier(shadow_dec_tree):
    assert shadow_dec_tree.is_classifier() == True


def test_class_weight(shadow_dec_tree):
    assert shadow_dec_tree.get_class_weight() is None


def test_criterion(shadow_dec_tree):
    assert shadow_dec_tree.criterion() == "GINI"


def test_nclasses(shadow_dec_tree):
    assert shadow_dec_tree.nclasses() == 2, "Number of classes should be 2"


def test_classes(shadow_dec_tree):
    assert shadow_dec_tree.classes()[0] == 0
    assert shadow_dec_tree.classes()[1] == 1


def test_get_node_samples(shadow_dec_tree):
    node_samples = shadow_dec_tree.get_node_samples()
    assert node_samples[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert node_samples[1] == [1, 2, 3, 8, 9, 10, 11, 14, 15, 18, 19]
    assert node_samples[2] == [1, 3, 9, 11, 15]
    assert node_samples[5] == [2, 8, 10, 14, 18, 19]
    assert node_samples[9] == [0, 4, 5, 7, 12, 13, 16, 17]


def test_get_class_weights(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_class_weights(), np.array([1, 1]))


def test_get_tree_nodes(shadow_dec_tree):
    assert [node.id for node in shadow_dec_tree.leaves] == [3, 4, 6, 7, 10, 11, 13, 14]
    assert [node.id for node in shadow_dec_tree.internal] == [2, 5, 1, 9, 12, 8, 0]


def test_get_children_left(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_left(),
                          np.array([1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1]))


def test_get_children_right(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_right(),
                          np.array([8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1]))


def test_get_node_split(shadow_dec_tree):
    assert shadow_dec_tree.get_node_split(0) == 0.5
    assert shadow_dec_tree.get_node_split(1) == 2.5
    assert shadow_dec_tree.get_node_split(3) == -2
    assert shadow_dec_tree.get_node_split(12) == 17.5


def test_get_node_feature(shadow_dec_tree):
    assert shadow_dec_tree.get_node_feature(0) == 3
    assert shadow_dec_tree.get_node_feature(2) == 1
    assert shadow_dec_tree.get_node_feature(4) == -2
    assert shadow_dec_tree.get_node_feature(8) == 4
    assert shadow_dec_tree.get_node_feature(12) == 1


def test_get_max_depth(shadow_dec_tree):
    assert shadow_dec_tree.get_max_depth() == 3, "Max depth should be 3"


def test_get_score(shadow_dec_tree):
    assert shadow_dec_tree.get_score() == 0.75, "Score should be 0.75"


def test_get_min_samples_leaf(shadow_dec_tree): \
    assert shadow_dec_tree.get_min_samples_leaf() == 1, "min_samples_leaf should be 1"

def test_nnodes(shadow_dec_tree):
    assert shadow_dec_tree.nnodes() == 15, "number of nodes should be 15"

