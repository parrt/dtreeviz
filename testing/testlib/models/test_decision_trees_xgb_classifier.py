import joblib
import numpy as np
import pytest
import xgboost as xgb

from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree


@pytest.fixture()
def xgb_booster() -> xgb.Booster:
    return joblib.load("fixtures/xgb_model_classifier.joblib")


@pytest.fixture()
def xgb_tree(xgb_booster, x_dataset_classifier, y_dataset_classifier) -> ShadowXGBDTree:
    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Survived"
    # class_names = list(dec_tree.classes_)
    return ShadowXGBDTree(xgb_booster, 1, x_dataset_classifier, y_dataset_classifier, features, target)


def test_x_dataset(x_dataset_classifier):
    dataset = xgb.DMatrix(x_dataset_classifier)
    assert dataset.num_row() == 20, "Number of rows should be 20"
    assert dataset.num_col() == 6, "Number of columns/features should be 6"


def test_y_dataset(y_dataset_classifier):
    assert y_dataset_classifier.shape[0] == 20, "Number of rows should be 20"
    assert list(y_dataset_classifier) == [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1]


def test_feature_names(xgb_booster):
    assert xgb_booster.feature_names == ['Pclass', 'Age', 'Fare', 'Sex_label', 'Cabin_label', 'Embarked_label']


def test_get_children_left(xgb_tree):
    assert np.array_equal(np.array(list(xgb_tree.get_children_left().values())), np.array([1, 3, 5, -1, -1, -1, -1]))
    assert not np.array_equal(np.array(list(xgb_tree.get_children_left().values())), np.array([-1, -1, -1, -1, 1, 3, 5]))


def test_get_children_right(xgb_tree):
    assert np.array_equal(np.array(list(xgb_tree.get_children_right().values())), np.array([2, 4, 6, -1, -1, -1, -1]))


def test_get_node_feature(xgb_tree):
    assert xgb_tree.get_node_feature(3) == -2
    assert xgb_tree.get_node_feature(1) == 0
    assert xgb_tree.get_node_feature(0) == 3
    assert xgb_tree.get_node_feature(6) == -2
    assert xgb_tree.get_node_feature(2) == 4


def test_get_features(xgb_tree):
    assert np.array_equal(np.array(list(xgb_tree.get_features().values())), np.array([3, 0, 4, -2, -2, -2, -2]))


def test_get_node_samples(xgb_tree):
    node_samples = xgb_tree.get_node_samples()
    assert node_samples[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert node_samples[1] == [1, 2, 3, 8, 9, 10, 11, 14, 15, 18, 19]
    assert node_samples[2] == [0, 4, 5, 6, 7, 12, 13, 16, 17]
    assert node_samples[3] == [1, 3, 9, 11, 15]
    assert node_samples[4] == [2, 8, 10, 14, 18, 19]
    assert node_samples[5] == [0, 4, 5, 7, 12, 13, 16, 17]
    assert node_samples[6] == [6]


def test_get_node_nsamples_by_class(xgb_tree):
    assert np.array_equal(xgb_tree.get_node_nsamples_by_class(0), np.array([10, 10]))
    assert np.array_equal(xgb_tree.get_node_nsamples_by_class(1), np.array([2, 9]))
    assert np.array_equal(xgb_tree.get_node_nsamples_by_class(2), np.array([8, 1]))
    assert np.array_equal(xgb_tree.get_node_nsamples_by_class(5), np.array([7, 1]))


def test_get_prediction(xgb_tree):
    assert xgb_tree.get_prediction(3) == 1
    assert xgb_tree.get_prediction(4) == 1
    assert xgb_tree.get_prediction(5) == 0
    assert xgb_tree.get_prediction(6) == 0


def test_nclasses(xgb_tree):
    assert xgb_tree.nclasses() == 2


def test_classes(xgb_tree):
    assert np.array_equal(xgb_tree.classes(), np.array([0, 1]))


def test_get_thresholds(xgb_tree):
    assert np.array_equal(np.array(list(xgb_tree.get_thresholds().values())), np.array([1, 3, 4, -2, -2, -2, -2]))


def test_is_classifier(xgb_tree):
    assert xgb_tree.is_classifier() == True


def test_get_leaf_sample_counts(xgb_tree):
    leaf_ids, leaf_samples = xgb_tree.get_leaf_sample_counts()
    assert np.array_equal(leaf_ids, np.array([3, 4, 5, 6])), "Leaf ids should be [3, 4, 5, 6]"
    assert np.array_equal(leaf_samples, np.array([5, 6, 8, 1])), "Leaf samples should be [5, 6, 8, 1]"
