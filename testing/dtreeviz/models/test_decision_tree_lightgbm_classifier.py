import lightgbm as lgb
import pytest
from dtreeviz.models.lightgbm_decision_tree import ShadowLightGBMTree
import numpy as np


@pytest.fixture()
def lgb_dec_tree():
    return lgb.Booster(model_file="fixtures/lightgbm_model_classifier.txt")


@pytest.fixture()
def shadow_dec_tree(lgb_dec_tree, dataset_lightgbm) -> ShadowLightGBMTree:
    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Survived"
    return ShadowLightGBMTree(lgb_dec_tree, 1, dataset_lightgbm[features], dataset_lightgbm[target], features, target)


def test_is_fit(shadow_dec_tree: ShadowLightGBMTree):
    assert shadow_dec_tree.is_fit() is True


def test_is_classifier(shadow_dec_tree: ShadowLightGBMTree):
    assert shadow_dec_tree.is_classifier() is True, "Should be a classifier decision tree"


def test_get_children_left(shadow_dec_tree: ShadowLightGBMTree):
    assert np.array_equal(shadow_dec_tree.get_children_left(), np.array(
        [1, 2, 3, 4, -1, -1, -1, 8, 9, -1, -1, -1, 13, 14, 15, -1, -1, 18, -1, -1, 21, 22, -1, -1, -1]))


def test_get_children_right(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_right(), np.array(
        [12, 7, 6, 5, -1, -1, -1, 11, 10, -1, -1, -1, 20, 17, 16, -1, -1, 19, -1, -1, 24, 23, -1, -1, -1]))
