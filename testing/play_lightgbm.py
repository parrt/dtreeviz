import sys
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import dtreeviz

random_state = 1234 # get reproducible trees

dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv"
dataset = pd.read_csv(dataset_url)
# Fill missing values for Age
dataset.fillna({"Age":dataset.Age.mean()}, inplace=True)
# Encode categorical variables
dataset["Sex"] = dataset.Sex.astype("category")#.cat.codes
dataset["Cabin"] = dataset.Cabin.astype("category").cat.codes
dataset.fillna({"Embarked":"?"}, inplace=True)
dataset["Embarked"] = dataset.Embarked.astype("category")#.cat.codes
print(dataset.dtypes)


features = ["Pclass", "Age", "Fare", "Sex", "Cabin", "Embarked"]
target = "Survived"

X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.2, random_state=42)

train_data = lgb.Dataset(data=X_train, label=y_train, feature_name=features, categorical_feature=["Sex", "Pclass", "Embarked"])
valid_data = lgb.Dataset(data=X_test, label=y_test, feature_name=features, categorical_feature=["Sex", "Pclass", "Embarked"])

lgbm_params = {
    'boosting': 'dart',          # dart (drop out trees) often performs better
    'application': 'binary',     # Binary classification
    'learning_rate': 0.05,       # Learning rate, controls size of a gradient descent step
    'min_data_in_leaf': 2,       # Data set is quite small so reduce this a bit
    'feature_pre_filter': False,
    'feature_fraction': 0.7,     # Proportion of features in each boost, controls overfitting
    'num_leaves': 41,            # Controls size of tree since LGBM uses leaf wise splits
    'drop_rate': 0.15,
    'max_depth':4,
    "seed":1212,
    'categorical_feature': 'auto'
}

lgbm_model = lgb.train(lgbm_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=False)


viz_model = dtreeviz.model(lgbm_model, tree_index=1,
                           X_train=dataset[features].values, y_train=dataset[target],
                           feature_names=features,
                           target_name=target, class_names=["survive", "perish"])

viz_model.view().show()
