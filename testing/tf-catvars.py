import tensorflow_decision_forests as tfdf
import dtreeviz
print(tfdf.__version__, dtreeviz.__version__ )

import tensorflow as tf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from matplotlib import pyplot as plt

def split_dataset(dataset, test_ratio=0.30):
  """
  Splits a panda dataframe in two, usually for train/test sets.
  Using the same random seed ensures we get the same split so
  that the description in this tutorial line up with generated images.
  """
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

np.random.seed(1234)

df_penguins = pd.read_csv("penguins.csv")
df_penguins = df_penguins.dropna()

penguin_label = "species"   # Name of the classification target label

classes = df_penguins[penguin_label].unique().tolist()
df_penguins[penguin_label] = df_penguins[penguin_label].map(classes.index)

penguin_features = list(df_penguins.columns)
print(f"Target '{penguin_label}'' classes: {classes}")

print(df_penguins.head(3))

train_ds_pd, test_ds_pd = split_dataset(df_penguins)
print(f"{len(train_ds_pd)} examples in training, {len(test_ds_pd)} examples for testing.")

# Convert to tensorflow data sets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=penguin_label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=penguin_label)

cmodel = tfdf.keras.RandomForestModel(num_trees=20, verbose=0, random_seed=1234)
cmodel.fit(x=train_ds)

cmodel.compile(metrics=["accuracy"])
print(cmodel.evaluate(test_ds, return_dict=True, verbose=0))

which_tree = 3 # pick a tree from the forest to visualize

viz_cmodel = dtreeviz.model(cmodel,
                           tree_index=which_tree,
                           X_train=train_ds_pd[penguin_features],
                           y_train=train_ds_pd[penguin_label],
                           feature_names=penguin_features,
                           target_name=penguin_label,
                           class_names=classes)

#viz_cmodel.view(leaftype='barh').show()
# viz_cmodel.view().show()
#viz_cmodel.view().show() # crashes in pie chart. can't figure out why

# viz_cmodel.ctree_feature_space(features=['island','flipper_length_mm'], show={'splits','legend'}, figsize=(5,5))

viz_cmodel.ctree_feature_space(features=['flipper_length_mm'], show={'splits','legend'}, figsize=(5,1.5))
plt.show()