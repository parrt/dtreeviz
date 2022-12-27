import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import (
    make_pipeline,
    Pipeline,
)
from sklearn.preprocessing import PolynomialFeatures

from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from dtreeviz.utils import (
    _extract_final_feature_names,
    extract_params_from_pipeline,
)

features_expected = [
    "1", "Pclass", "Age", "Fare", "Cabin_label", "Embarked_label",
    "Pclass Age", "Pclass Fare", "Pclass Cabin_label", "Pclass Embarked_label",
    "Age Fare", "Age Cabin_label", "Age Embarked_label", "Fare Cabin_label",
    "Fare Embarked_label", "Cabin_label Embarked_label"
]


@pytest.fixture()
def dec_tree_pipeline(x_dataset_classifier, y_dataset_classifier) -> (Pipeline):

    model = make_pipeline(
        VarianceThreshold(threshold=0.5),
        PolynomialFeatures(degree=2, interaction_only=True),
        DecisionTreeClassifier(max_depth=3))
    model.fit(x_dataset_classifier, y_dataset_classifier)
    return model


@pytest.fixture()
def shadow_dec_tree(dec_tree_pipeline, dataset) -> ShadowSKDTree:
    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Survived"

    tree_model, X_train, feature_names = extract_params_from_pipeline(
        pipeline=dec_tree_pipeline,
        X_train=dataset[features],
        feature_names=features
    )

    return ShadowSKDTree(
        tree_model=tree_model,
        X_train=X_train,
        y_train=dataset[target],
        feature_names=feature_names,
        class_names=list(dec_tree_pipeline.classes_)
    )


def test_extract_feature_names(dec_tree_pipeline):
    features_actual = _extract_final_feature_names(
        pipeline=dec_tree_pipeline,
        features=np.array(["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"])
    )
    features_expected = [
        "1", "Pclass", "Age", "Fare", "Cabin_label", "Embarked_label",
        "Pclass Age", "Pclass Fare", "Pclass Cabin_label", "Pclass Embarked_label",
        "Age Fare", "Age Cabin_label", "Age Embarked_label", "Fare Cabin_label",
        "Fare Embarked_label", "Cabin_label Embarked_label"
    ]
    assert features_actual == features_expected


def test_feature_number(shadow_dec_tree):
    assert shadow_dec_tree.feature_names == features_expected


def test_is_fit(shadow_dec_tree):
    assert shadow_dec_tree.is_fit()
