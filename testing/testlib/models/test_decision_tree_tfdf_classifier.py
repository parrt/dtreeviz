import pytest
import numpy as np

import tensorflow_decision_forests as tfdf
from dtreeviz.models.tfdf_decision_tree import ShadowTFDFTree


@pytest.fixture()
def features_clf():
    return ["Pclass", "Sex_label", "Embarked_label", "Age_mean", "SibSp", "Parch", "Fare"]


@pytest.fixture()
def tfdf_rf_model(dataset_spark_tf):
    """
    The saved model is not the same with the raw one. When we load it, it is an instance of a _RandomForestInspector class
    Because of this, it would be better to recreate the same model and to be used in these unit tests.

    :param dataset_spark_tf:
    :return:
    """

    random_state = 1234
    target_clf = "Survived"
    train_clf = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_spark_tf, label=target_clf)
    model_clf = tfdf.keras.RandomForestModel(max_depth=3, random_seed=random_state)
    model_clf.fit(train_clf)

    return model_clf


@pytest.fixture()
def tfdf_shadow_clf(tfdf_rf_model, dataset_spark_tf, features_clf):
    target_clf = "Survived"

    tfdf_shadow = ShadowTFDFTree(tfdf_rf_model,
                                 tree_index=0,
                                 x_data=dataset_spark_tf[features_clf],
                                 y_data=dataset_spark_tf[target_clf],
                                 feature_names=features_clf,
                                 target_name=target_clf,
                                 class_names=[0, 1])

    return tfdf_shadow


def test_is_fit(tfdf_shadow_clf):
    assert tfdf_shadow_clf.is_fit() is True


def test_get_children_left(tfdf_shadow_clf):
    assert tfdf_shadow_clf.get_children_left() == {0: 1, 1: 2, 4: 5, 2: -1, 3: -1, 5: -1, 6: -1}


def test_get_children_right(tfdf_shadow_clf):
    assert tfdf_shadow_clf.get_children_right() == {1: 3, 0: 4, 4: 6, 2: -1, 3: -1, 5: -1, 6: -1}
