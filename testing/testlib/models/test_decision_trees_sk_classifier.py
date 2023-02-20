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
    assert shadow_dec_tree.criterion() == "Gini"


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


def test_get_leaf_sample_counts(shadow_dec_tree):
    leaf_ids, leaf_samples = shadow_dec_tree.get_leaf_sample_counts()
    assert np.array_equal(leaf_ids,
                          np.array([3, 4, 6, 7, 10, 11, 13, 14])), "Leaf ids should be [3, 4, 6, 7, 10, 11, 13, 14]"
    assert np.array_equal(leaf_samples,
                          np.array([0, 5, 6, 0, 2, 6, 0, 1])), "Leaf samples should be [0, 5, 6, 0, 2, 6, 0, 1]"


def test_get_thresholds(shadow_dec_tree):
    assert list(shadow_dec_tree.get_thresholds()) == [0.5, 2.5, 2.5, -2.0, -2.0, 23.350000381469727, -2.0, -2.0, 3.5,
                                                      3.5, -2.0, -2.0, 17.5, -2.0, -2.0]


def test_predict(shadow_dec_tree, x_dataset_classifier):
    leaf_pred_0 = shadow_dec_tree.predict(x_dataset_classifier.iloc[0])
    assert leaf_pred_0 == 0

    leaf_pred_2 = shadow_dec_tree.predict(x_dataset_classifier.iloc[2])
    assert leaf_pred_2 == 1

    leaf_pred_6 = shadow_dec_tree.predict(x_dataset_classifier.iloc[6])
    assert leaf_pred_6 == 0

    leaf_pred_9 = shadow_dec_tree.predict(x_dataset_classifier.iloc[9])
    assert leaf_pred_9 == 1

    leaf_pred_7 = shadow_dec_tree.predict(x_dataset_classifier.iloc[7])
    assert leaf_pred_7 == 1


def test_predict_path(shadow_dec_tree, x_dataset_classifier):
    def get_node_ids(nodes):
        return [node.id for node in nodes]

    leaf_pred_path_0 = shadow_dec_tree.predict_path(x_dataset_classifier.iloc[0])
    assert get_node_ids(leaf_pred_path_0) == [0, 8, 9, 11]

    leaf_pred_path_2 = shadow_dec_tree.predict_path(x_dataset_classifier.iloc[2])
    assert get_node_ids(leaf_pred_path_2) == [0, 1, 5, 6]

    leaf_pred_path_6 = shadow_dec_tree.predict_path(x_dataset_classifier.iloc[6])
    assert get_node_ids(leaf_pred_path_6) == [0, 8, 12, 14]

    leaf_pred_path_9 = shadow_dec_tree.predict_path(x_dataset_classifier.iloc[9])
    assert get_node_ids(leaf_pred_path_9) == [0, 1, 2, 4]

    leaf_pred_path_7 = shadow_dec_tree.predict_path(x_dataset_classifier.iloc[7])
    assert get_node_ids(leaf_pred_path_7) == [0, 8, 9, 10]

def test_get_prediction(shadow_dec_tree):
    assert shadow_dec_tree.get_prediction(3) == 0, "Prediction for leaf=3 should be 0"
    assert shadow_dec_tree.get_prediction(4) == 1, "Prediction for leaf=4 should be 1"
    assert shadow_dec_tree.get_prediction(6) == 1, "Prediction for leaf=6 should be 1"
    assert shadow_dec_tree.get_prediction(7) == 0, "Prediction for leaf=7 should be 0"
    assert shadow_dec_tree.get_prediction(10) == 1, "Prediction for leaf=10 should be 1"
    assert shadow_dec_tree.get_prediction(11) == 0, "Prediction for leaf=11 should be 0"
    assert shadow_dec_tree.get_prediction(13) == 1, "Prediction for leaf=13 should be 1"
    assert shadow_dec_tree.get_prediction(14) == 0, "Prediction for leaf=14 should be 0"


def test_get_node_nsamples_by_class(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(0), np.array([549, 342]))
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(1), np.array([81, 233]))
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(3), np.array([1, 1]))
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(5), np.array([72, 72]))
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(10), np.array([5, 9]))
    assert np.array_equal(shadow_dec_tree.get_node_nsamples_by_class(11), np.array([404, 55]))

# def test_get_split_samples(shadow_dec_tree):
#     hard to test because dataset is not the same with dataset the decision tree model was trained
#     left_0, right_0 = shadow_dec_tree.get_split_samples(0)
#     assert len(left_0) == 314 and len(right_0) == 577
