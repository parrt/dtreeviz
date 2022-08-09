import pandas as pd
import pytest
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"ROOT_DIR {ROOT_DIR}")


@pytest.fixture(autouse=True)
def dataset() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")


@pytest.fixture(autouse=True)
def x_dataset_classifier() -> pd.DataFrame:
    return pd.read_csv(f"{ROOT_DIR}/fixtures/dataset.csv")[
        ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]]


@pytest.fixture()
def y_dataset_classifier() -> pd.Series:
    return pd.read_csv(f"{ROOT_DIR}/fixtures/dataset.csv")["Survived"]


@pytest.fixture()
def x_dataset_regressor() -> pd.DataFrame:
    return pd.read_csv(f"{ROOT_DIR}/fixtures/dataset.csv")[
        ["Pclass", "Survived", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]]


@pytest.fixture()
def y_dataset_regressor(dataset) -> pd.Series:
    return dataset["Age"]


@pytest.fixture()
def dataset_spark_tensorflow() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset_spark_tf.csv")

@pytest.fixture()
def dataset_lightgbm() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset_lightgbm.csv")