import numpy as np
from animl.trees import *
from animl.viz.trees import *

parrt_article = "/Users/parrt/github/ml-articles/decision-tree-viz/images"

def viz_boston_one_feature(orientation="TD", max_depth=3, random_state=666, fancy=True):
    regr = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    boston = load_boston()

    i = 6
    X_train = boston.data[:, i].reshape(-1, 1)
    y_train = boston.target
    regr.fit(X_train, y_train)

    viz = dtreeviz(regr, X_train, y_train, target_name='price',
                   feature_names=[boston.feature_names[i]], orientation=orientation,
                   fancy=fancy,
                   show_node_labels=True,
                   X=None)

    viz.save(f"{parrt_article}/boston-TD-AGE.svg")

def viz_knowledge_one_feature(orientation="TD", max_depth=3, random_state=666, fancy=True):
    # data from https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    know = pd.read_csv("data/knowledge.csv")
    target_names = ['very_low', 'Low', 'Middle', 'High']
    know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(target_names)})

    the_feature = "PEG"
    X_train, y_train = know[[the_feature]], know['UNS']
    clf.fit(X_train, y_train)

    X = X_train.iloc[np.random.randint(0, len(X_train))]
    X = None

    viz = dtreeviz(clf, X_train, y_train, target_name='UNS',
                  feature_names=[the_feature], orientation=orientation,
                  class_names=target_names,
                  show_node_labels = True,
                  fancy=fancy,
                  X=X)

    viz.save(f"{parrt_article}/knowledge-TD-PEG.svg")

viz_boston_one_feature(fancy=True, max_depth=3, orientation='TD')
viz_knowledge_one_feature(fancy=True, orientation='TD', max_depth=3)
