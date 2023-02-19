import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd

import dtreeviz

np.random.seed(2)

from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
features = list(iris.feature_names)

features = [f.replace(' (cm)','').replace(' ', '_') for f in features]
classes = iris.target_names
label = 'iris'
df = pd.DataFrame(iris.data, columns=features)

features += ['state']
states = ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
          'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts',
          'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
          'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
          'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
          'washington', 'west virginia', 'wisconsin', 'wyoming']
smap = {i:s for i,s in enumerate(states)}
print(smap)
df[label] = iris.target

df['state'] = np.random.randint(0, 50, 150)
df['state'] = df['state'].map(smap)
#df.loc[(df['iris']==1)&(df['petal_length']<3.0), 'state'] = 10
df.loc[df['iris']==1, 'state'] = 'utah'

print(features, classes)
# print(df['state'].unique())
# print(df)

m = tfdf.keras.RandomForestModel(num_trees=1, verbose=0, random_seed=1234)
tensors = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label)
m.fit(x=tensors)

v = dtreeviz.model(m,
                   tree_index=0,
                   X_train=df[features],
                   y_train=df[label],
                   feature_names=features,
                   target_name=label,
                   class_names=list(classes))

v.view().show()
plt.show()