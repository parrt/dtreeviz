import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd

import dtreeviz

np.random.seed(1)

def split_dataset(dataset, test_ratio=0.30):
  """
  Splits a panda dataframe in two, usually for train/test sets.
  Using the same random seed ensures we get the same split so
  that the description in this tutorial line up with generated images.
  """
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

df_abalone = pd.read_csv("abalone.csv")

abalone_label = "Rings"   # Name of the classification target label
abalone_features = list(df_abalone.columns)

# Split into training and test sets 70/30
df_train_abalone, df_test_abalone = split_dataset(df_abalone)
print(f"{len(df_train_abalone)} examples in training, {len(df_test_abalone)} examples for testing.")

# Convert to tensorflow data sets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train_abalone, label=abalone_label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test_abalone, label=abalone_label, task=tfdf.keras.Task.REGRESSION)

rmodel = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION,
                                      max_depth=5,      # don't let the tree get too big
                                      random_seed=1234, # create same tree every time
                                      verbose=0)
rmodel.fit(x=train_ds)

which_tree = 2
viz_rmodel = dtreeviz.model(rmodel, tree_index=which_tree,
                           X_train=df_train_abalone[abalone_features],
                           y_train=df_train_abalone[abalone_label],
                           feature_names=abalone_features,
                           target_name='Rings')

viz_rmodel.view().show()