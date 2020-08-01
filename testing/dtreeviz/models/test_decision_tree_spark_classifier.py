import pytest
from pyspark.ml.classification import DecisionTreeClassificationModel
from dtreeviz.models.spark_decision_tree import ShadowSparkTree
from pyspark.sql import SparkSession
import numpy as np


@pytest.fixture()
def tree_model() -> (DecisionTreeClassificationModel):
    SparkSession.builder \
        .master("local[2]") \
        .appName("dtreeviz_sparkml") \
        .getOrCreate()
    return DecisionTreeClassificationModel.load("fixtures/spark_decision_tree_classifier.model")


@pytest.fixture()
def shadow_dec_tree(tree_model, dataset_spark) -> ShadowSparkTree:
    features = ["Pclass", "Sex_label", "Embarked_label", "Age_mean", "SibSp", "Parch", "Fare"]
    target = "Survived"
    return ShadowSparkTree(tree_model, dataset_spark[features], dataset_spark[target], features, target)


def test_is_fit(shadow_dec_tree):
    assert shadow_dec_tree.is_fit() is True


def test_get_children_left(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_left(),
                          np.array([1, 2, 3, -1, -1, -1, 7, 8, 9, -1, -1, -1, 13, 14, -1, -1, -1]))


def test_get_children_right(shadow_dec_tree):
    assert np.array_equal(shadow_dec_tree.get_children_right(),
                          np.array([6, 5, 4, -1, -1, -1, 12, 11, 10, -1, -1, -1, 16, 15, -1, -1, -1]))
