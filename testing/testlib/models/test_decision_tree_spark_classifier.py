import numpy as np
import pyspark
import pytest
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql import SparkSession

from dtreeviz.models.spark_decision_tree import ShadowSparkTree


@pytest.fixture()
def tree_model() -> (DecisionTreeClassificationModel):
    SparkSession.builder \
        .master("local[2]") \
        .appName("dtreeviz_sparkml") \
        .getOrCreate()

    spark_major_version = int(pyspark.__version__.split(".")[0])
    if spark_major_version >= 3:
        return DecisionTreeClassificationModel.load("fixtures/spark_3_0_decision_tree_classifier.model")
    elif spark_major_version >= 2:
        return DecisionTreeClassificationModel.load("fixtures/spark_2_decision_tree_classifier.model")


@pytest.fixture()
def spark_dtree(tree_model, dataset_spark_tensorflow) -> ShadowSparkTree:
    features = ["Pclass", "Sex_label", "Embarked_label", "Age_mean", "SibSp", "Parch", "Fare"]
    target = "Survived"
    return ShadowSparkTree(tree_model, dataset_spark_tensorflow[features], dataset_spark_tensorflow[target], features, target)


def test_is_fit(spark_dtree):
    assert spark_dtree.is_fit() is True


def test_is_classifier(spark_dtree):
    assert spark_dtree.is_classifier() is True, "Spark decision tree should be classifier"


def test_get_children_left(spark_dtree):
    assert np.array_equal(spark_dtree.get_children_left(),
                          np.array([1, 2, 3, -1, -1, -1, 7, 8, 9, -1, -1, -1, 13, 14, -1, -1, -1]))


def test_get_children_right(spark_dtree):
    assert np.array_equal(spark_dtree.get_children_right(),
                          np.array([6, 5, 4, -1, -1, -1, 12, 11, 10, -1, -1, -1, 16, 15, -1, -1, -1]))


def test_get_node_nsamples(spark_dtree):
    assert spark_dtree.get_node_nsamples(0) == 891, "Node samples for node 0 should be 891"
    assert spark_dtree.get_node_nsamples(1) == 577, "Node samples for node 1 should be 577"
    assert spark_dtree.get_node_nsamples(5) == 559, "Node samples for node 5 should be 559"
    assert spark_dtree.get_node_nsamples(8) == 3, "Node samples for node 3 should be 3"
    assert spark_dtree.get_node_nsamples(12) == 144, "Node samples for node 12 should be 144"
    assert spark_dtree.get_node_nsamples(10) == 2, "Node samples node node 10 should be 2"
    assert spark_dtree.get_node_nsamples(16) == 23, "Node samples for node 16 should be 23"


def test_get_features(spark_dtree):
    assert np.array_equal(spark_dtree.get_features(),
                          np.array([1, 3, 4, -1, -1, -1, 0, 3, 0, -1, -1, -1, 6, 2, -1, -1,
                                    -1])), "Feature indexes should be [1, 3, 4, -1, -1, -1, 0, 3, 0, -1, -1, -1, 6, 2, -1, -1, -1]"


def test_nclasses(spark_dtree):
    assert spark_dtree.nclasses() == 2, "n classes should be 2"


def test_get_node_feature(spark_dtree):
    assert spark_dtree.get_node_feature(0) == 1, "Feature index for node 0 should be 1"
    assert spark_dtree.get_node_feature(1) == 3, "Feature index for node 1 should be 3"
    assert spark_dtree.get_node_feature(3) == -1, "Feature index for node 3 should be -1"
    assert spark_dtree.get_node_feature(7) == 3, "Feature index for node 7 should be 3"
    assert spark_dtree.get_node_feature(8) == 0, "Feature index for node 8 should be 0"
    assert spark_dtree.get_node_feature(12) == 6, "Feature index for node 12 should be 6"
    assert spark_dtree.get_node_feature(16) == -1, "Feature index for node 16 should be -1"


def test_get_node_nsamples_by_class(spark_dtree):
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(0), np.array([549.0, 342.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(1), np.array([468.0, 109.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(5), np.array([463.0, 96.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(8), np.array([1.0, 2.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(11), np.array([8.0, 159.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(13), np.array([51.0, 70.0]))
    assert np.array_equal(spark_dtree.get_node_nsamples_by_class(16), np.array([21.0, 2.0]))


def test_get_prediction(spark_dtree):
    assert spark_dtree.get_prediction(0) == 0, "Prediction value for node 0 should be 0"
    assert spark_dtree.get_prediction(1) == 0, "Prediction value for node 1 should be 0"
    assert spark_dtree.get_prediction(4) == 0, "Prediction value for node 4 should be 0"
    assert spark_dtree.get_prediction(6) == 1, "Prediction value for node 6 should be 1"
    assert spark_dtree.get_prediction(8) == 1, "Prediction value for node 8 should be 1"
    assert spark_dtree.get_prediction(10) == 1, "Prediction value for node 10 should be 1"
    assert spark_dtree.get_prediction(12) == 0, "Prediction value for node 12 should be 0"
    assert spark_dtree.get_prediction(14) == 1, "Prediction value for node 14 should be 1"
    assert spark_dtree.get_prediction(15) == 0, "Prediction value for node 15 should be 0"


def test_nnodes(spark_dtree):
    assert spark_dtree.nnodes() == 17, "Number of nodes from tree should be 17"


def test_get_max_depth(spark_dtree):
    assert spark_dtree.get_max_depth() == 4, "Max depth should be 4"


def test_get_min_samples_leaf(spark_dtree):
    assert spark_dtree.get_min_samples_leaf() == 1, "Min sample leaf should be 1"


def test_get_thresholds(spark_dtree):
    assert np.array_equal(spark_dtree.get_thresholds(),
                          np.array([list([0.0]), 3.5, 2.5, -1, -1, -1, 2.5, 3.5, 1.5, -1, -1, -1,
                                    24.808349999999997,
                                    list([1.0, 2.0]), -1, -1, -1]))


def test_predict(spark_dtree, dataset_spark_tensorflow):
    leaf_pred_0 = spark_dtree.predict(dataset_spark_tensorflow.iloc[0])
    assert leaf_pred_0 == 0

    leaf_pred_10 = spark_dtree.predict(dataset_spark_tensorflow.iloc[10])
    assert leaf_pred_10 == 0

    leaf_pred_109 = spark_dtree.predict(dataset_spark_tensorflow.iloc[109])
    assert leaf_pred_109 == 1

    leaf_pred_119 = spark_dtree.predict(dataset_spark_tensorflow.iloc[119])
    assert leaf_pred_119 == 0


def test_predict_path(spark_dtree, dataset_spark_tensorflow):
    def get_node_ids(nodes):
        return [node.id for node in nodes]

    leaf_pred_path_0 = spark_dtree.predict_path(dataset_spark_tensorflow.iloc[0])
    assert get_node_ids(leaf_pred_path_0) == [0, 1, 5]

    leaf_pred_path_10 = spark_dtree.predict_path(dataset_spark_tensorflow.iloc[10])
    assert get_node_ids(leaf_pred_path_10) == [0, 6, 12, 13, 15]

    leaf_pred_path_109 = spark_dtree.predict_path(dataset_spark_tensorflow.iloc[109])
    assert get_node_ids(leaf_pred_path_109) == [0, 6, 12, 13, 14]

    leaf_pred_path_119 = spark_dtree.predict_path(dataset_spark_tensorflow.iloc[119])
    assert get_node_ids(leaf_pred_path_119) == [0, 6, 12, 16]


def test_get_node_samples(spark_dtree):
    assert len(spark_dtree.get_node_samples()[0]) == 891
    assert len(spark_dtree.get_node_samples()[2]) == 18
    assert len(spark_dtree.get_node_samples()[5]) == 559
    assert len(spark_dtree.get_node_samples()[11]) == 167
    assert len(spark_dtree.get_node_samples()[13]) == 121
    assert len(spark_dtree.get_node_samples()[16]) == 23


def test_is_categorical_split(spark_dtree):
    assert spark_dtree.is_categorical_split(1) is False
    assert spark_dtree.is_categorical_split(0) is True
    assert spark_dtree.is_categorical_split(5) is False
    assert spark_dtree.is_categorical_split(12) is False
    assert spark_dtree.is_categorical_split(13) is True
