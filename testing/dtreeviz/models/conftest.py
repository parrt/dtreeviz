import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def dataset() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")


@pytest.fixture(autouse=True)
def x_dataset() -> pd.DataFrame:
    return pd.read_csv("fixtures/dataset.csv")[["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]]
