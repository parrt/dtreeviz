import pytest
import joblib
import xgboost as xgb
from dtreeviz.models.xgb_decision_tree import XGBDTree
import numpy as np
import pandas as pd


# @pytest.fixture()
# def x_dataset() -> pd.DataFrame:
#     return pd.read_csv("fixtures/dataset.csv")[["Pclass","Age","Fare","Sex_label","Cabin_label","Embarked_label"]]
#

@pytest.fixture()
def xgb_booster() -> xgb.Booster:
    return joblib.load("fixtures/xgb_model.joblib")


@pytest.fixture()
def xgb_tree(x_dataset, xgb_booster) -> XGBDTree:
    return XGBDTree(xgb_booster, 1)


def test_x_dataset(x_dataset):
    dataset = xgb.DMatrix(x_dataset)
    assert dataset.num_row() == 20, "Number of rows should be 20"
    assert dataset.num_col() == 6, "Number of columns/features should be 6"


def test_feature_names(xgb_booster):
    assert xgb_booster.feature_names == ['Pclass', 'Age', 'Fare', 'Sex_label', 'Cabin_label', 'Embarked_label']


def test_get_children_left(xgb_tree):
    assert np.array_equal(xgb_tree.get_children_left(), np.array([1, 3, 5, -1, -1, -1, -1]))
    assert not np.array_equal(xgb_tree.get_children_left(), np.array([-1, -1, -1, -1, 1, 3, 5]))


def test_get_right_children_left(xgb_tree):
    assert np.array_equal(xgb_tree.get_children_right(), np.array([2, 4, 6, -1, -1, -1, -1]))


def test_get_node_feature(xgb_tree):
    assert xgb_tree.get_node_feature(3) == -2
    assert xgb_tree.get_node_feature(1) == 0
    assert xgb_tree.get_node_feature(0) == 3
    assert xgb_tree.get_node_feature(6) == -2
    assert xgb_tree.get_node_feature(2) == 4


def test_get_node_samples(xgb_tree, x_dataset):
    node_samples = xgb_tree.get_node_samples(x_dataset)
    assert node_samples[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert node_samples[1] == [1, 2, 3, 8, 9, 10, 11, 14, 15, 18, 19]
    assert node_samples[2] == [0, 4, 5, 6, 7, 12, 13, 16, 17]
    assert node_samples[3] == [1, 3, 9, 11, 15]
    assert node_samples[4] == [2, 8, 10, 14, 18, 19]
    assert node_samples[5] == [0, 4, 5, 7, 12, 13, 16, 17]
    assert node_samples[6] == [6]
