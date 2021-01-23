import joblib
import numpy as np
import pytest
import xgboost as xgb

from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree


@pytest.fixture()
def xgb_booster() -> xgb.Booster:
    return joblib.load("fixtures/xgb_model_regressor.joblib")


@pytest.fixture()
def xgb_tree(xgb_booster, x_dataset_regressor, y_dataset_regressor) -> ShadowXGBDTree:
    features = ["Pclass", "Survived", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Age"
    return ShadowXGBDTree(xgb_booster, 1, x_dataset_regressor, y_dataset_regressor, features, target)


def test_x_dataset(x_dataset_regressor):
    dataset = xgb.DMatrix(x_dataset_regressor)
    assert dataset.num_row() == 20, "Number of rows should be 20"
    assert dataset.num_col() == 6, "Number of columns/features should be 6"


def test_y_dataset(y_dataset_regressor):
    assert y_dataset_regressor.shape[0] == 20, "Number of rows should be 20"
    assert list(y_dataset_regressor) == [22.0, 38.0, 26.0, 35.0, 35.0, 29.69911764705882, 54.0, 2.0, 27.0, 14.0, 4.0,
                                         58.0, 20.0, 39.0, 14.0, 55.0, 2.0, 29.69911764705882, 31.0, 29.69911764705882]


def test_feature_names(xgb_booster):
    assert xgb_booster.feature_names == ["Pclass", "Survived", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]


def test_get_prediction(xgb_tree):
    assert xgb_tree.get_prediction(0) == 28.254867647058823
    assert xgb_tree.get_prediction(1) == 29.531439628482975
    assert xgb_tree.get_prediction(2) == 4.0
    assert xgb_tree.get_prediction(3) == 40.52844537815126
    assert xgb_tree.get_prediction(4) == 23.116519607843134
    assert xgb_tree.get_prediction(5) == 26.04424836601307
    assert xgb_tree.get_prediction(6) == 14.333333333333334


def test_get_max_depth(xgb_tree):
    assert xgb_tree.get_max_depth() == 3, "max_depth should be 3"


def test_get_leaf_sample_counts(xgb_tree):
    leaf_ids, leaf_samples = xgb_tree.get_leaf_sample_counts()
    assert np.array_equal(leaf_ids, np.array([3, 5, 6, 2])), "Leaf ids should be [3, 5, 6, 2]"
    assert np.array_equal(leaf_samples, np.array([7, 9, 3, 1])), "Leaf samples should be [7, 9, 3, 1]"
