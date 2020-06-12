import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def dataset() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")


@pytest.fixture(autouse=True)
def x_dataset_classifier() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")[["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]]


@pytest.fixture()
def y_dataset_classifier() -> pd.Series:
    return pd.read_csv("fixtures/dataset.csv")["Survived"]


@pytest.fixture()
def x_dataset_regressor() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")[
        ["Pclass", "Survived", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]]


@pytest.fixture()
def y_dataset_regressor(dataset) -> pd.Series:
    return dataset["Age"]
