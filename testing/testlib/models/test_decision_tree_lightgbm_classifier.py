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


def test_get_node_nsamples(shadow_dec_tree):
    assert shadow_dec_tree.get_node_nsamples(0) == 712
    assert shadow_dec_tree.get_node_nsamples(3) == 110
    assert shadow_dec_tree.get_node_nsamples(4) == 56
    assert shadow_dec_tree.get_node_nsamples(7) == 115
    assert shadow_dec_tree.get_node_nsamples(13) == 340
    assert shadow_dec_tree.get_node_nsamples(19) == 175
    assert shadow_dec_tree.get_node_nsamples(21) == 86


def test_get_thresholds(shadow_dec_tree: ShadowLightGBMTree):
    thresholds = shadow_dec_tree.get_thresholds()
    assert thresholds[0] == 0
    assert thresholds[1] == 2.5
    assert thresholds[6] == -1
    assert thresholds[8] == 1.5
    assert thresholds[13] == 7.91
    assert thresholds[17] == 1.5
    assert thresholds[22] == -1


def test_nnodes(shadow_dec_tree):
    assert shadow_dec_tree.nnodes() == 25


def test_get_features(shadow_dec_tree: ShadowLightGBMTree):
    assert np.array_equal(shadow_dec_tree.get_features(), np.array(
        [3, 0, 2, 2, -1, -1, -1, 2, 5, -1, -1, -1, 2, 2, 2, -1, -1, 5, -1, -1, 0, 2, -1, -1, -1]))


def test_nclasses(shadow_dec_tree: ShadowLightGBMTree):
    assert shadow_dec_tree.nclasses() == 2


def test_classes(shadow_dec_tree: ShadowLightGBMTree):
    assert np.array_equal(shadow_dec_tree.classes(), np.array([0, 1]))


def test_get_node_samples(shadow_dec_tree: ShadowLightGBMTree):
    node_samples = shadow_dec_tree.get_node_samples();
    assert len(node_samples[0]) == 712
    assert len(node_samples[1]) == 245
    assert len(node_samples[8]) == 95
    assert len(node_samples[3]) == 110
    assert len(node_samples[13]) == 340
    assert len(node_samples[17]) == 196
    assert len(node_samples[21]) == 86


def test_get_split_samples(shadow_dec_tree: ShadowLightGBMTree):
    left_0, rigth_0 = shadow_dec_tree.get_split_samples(0)
    assert len(left_0) == 245 and len(rigth_0) == 467

    left_1, right_1 = shadow_dec_tree.get_split_samples(1)
    assert len(left_1) == 130 and len(right_1) == 115

    left_3, right_3 = shadow_dec_tree.get_split_samples(3)
    assert len(left_3) == 56 and len(right_3) == 54

    left_7, right_7 = shadow_dec_tree.get_split_samples(7)
    assert len(left_7) == 95 and len(right_7) == 20

    left_12, right_12 = shadow_dec_tree.get_split_samples(12)
    assert len(left_12) == 340 and len(right_12) == 127

    left_17, right_17 = shadow_dec_tree.get_split_samples(17)
    assert len(left_17) == 21 and len(right_17) == 175


def test_get_min_samples_leaf(shadow_dec_tree: ShadowLightGBMTree):
    assert shadow_dec_tree.get_min_samples_leaf() == 20
