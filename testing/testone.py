import graphviz
from dtreeviz.shadow import *
from gen_samples import *
import tempfile
from sklearn.tree import export_graphviz


def viz_iris(orientation="TD", max_depth=5, random_state=666, fancy=True):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    iris = load_iris()

    data = pd.DataFrame(iris.data)
    data.columns = iris.feature_names

    clf = clf.fit(data, iris.target)

    # for i in range(len(iris.data)):
    for i in [60]:
        x = data.iloc[i]
        pred = clf.predict([x.values])

        shadow_tree = ShadowDecTree(clf, iris.data, iris.target,
                                    feature_names=iris.feature_names, class_names=["setosa", "versicolor", "virginica"])

        pred2 = shadow_tree.predict(x.values)
        print(f'{x} -> {pred[0]} vs mine {pred2[0]}, path = {[f"node{p.feature_name()}" for p in pred2[1]]}')
        path = [n.id for n in pred2[1]]
        if pred[0]!=pred2[0]:
            print("MISMATCH!")

    features = list(data.columns)
    features = np.array(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    st = dtreeviz(clf, iris.data, iris.target, target_name='variety',
                  feature_names=features, orientation=orientation,
                  class_names=["setosa", "versicolor", "virginica"],  # 0,1,2 targets
                  #histtype='strip',
                  fancy=fancy,
                  X=x)

    return st

def viz_boston(orientation="TD", max_depth=3, random_state=666, fancy=True):
    regr = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    boston = load_boston()

    regr = regr.fit(boston.data, boston.target)

    X = boston.data[np.random.randint(0, len(boston.data)),:]

    print(boston.feature_names)
    features = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    viz = dtreeviz(regr, boston.data, boston.target, target_name='price',
                   feature_names=features, orientation=orientation,
                   fancy=fancy,
                   X=X)

    export_graphviz(regr, out_file="/tmp/boston-scikit-tree.dot",
                    filled=True, rounded=True,
                    special_characters=True)

    return viz

def viz_knowledge(orientation="TD", max_depth=3, random_state=666, fancy=True):
    # data from https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    know = pd.read_csv("data/knowledge.csv")
    target_names = ['very_low', 'Low', 'Middle', 'High']
    know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(target_names)})

    X_train, y_train = know.drop('UNS', axis=1), know['UNS']
    clf = clf.fit(X_train[['PEG','LPR']], y_train)

    X = X_train.iloc[np.random.randint(0, len(know))]

    viz = dtreeviz(clf, X_train[['PEG','LPR']], y_train, target_name='UNS',
                  feature_names=['PEG','LPR'], orientation=orientation,
                  class_names=target_names,
#                   show_node_labels=True,
                   histtype='strip',
                   fancy=fancy)
    return viz

def viz_diabetes(orientation="TD", max_depth=3, random_state=666, fancy=True, pickX=False):
    diabetes = load_diabetes()

    regr = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    regr.fit(diabetes.data, diabetes.target)

    X = None
    if pickX:
        X = diabetes.data[np.random.randint(0, len(diabetes.data)),:]

    viz = dtreeviz(regr, diabetes.data, diabetes.target, target_name='progr',
                  feature_names=diabetes.feature_names, orientation=orientation,
                  fancy=fancy,
                  X=X)

    return viz
#
# def viz_knowledge(orientation="TD", max_depth=3, random_state=666, fancy=True, pickX=False):
#     # data from https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
#     clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
#     know = pd.read_csv("data/knowledge.csv")
#     target_names = ['very_low', 'Low', 'Middle', 'High']
#     know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(target_names)})
#
#     X_train, y_train = know.drop('UNS', axis=1), know['UNS']
#     clf = clf.fit(X_train, y_train)
#
#     st = dtreeviz(clf, X_train, y_train, target_name='UNS',
#                   feature_names=X_train.columns.values, orientation=orientation,
#                   class_names=target_names,
#                   fancy=fancy,
#                   X=X_train.iloc[3,:])
#     return st

def viz_digits(orientation="TD", max_depth=3, random_state=666, fancy=True, pickX=False):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    digits = load_digits()

    # "8x8 image of integer pixels in the range 0..16."
    columns = [f'pixel[{i},{j}]' for i in range(8) for j in range(8)]

    clf.fit(digits.data, digits.target)

    X = None
    if pickX:
        X = digits.data[np.random.randint(0, len(digits.data)),:]

    viz = dtreeviz(clf, digits.data, digits.target, target_name='number',
                  feature_names=columns, orientation=orientation,
                  class_names=[chr(c) for c in range(ord('0'),ord('9')+1)],
                  fancy=fancy, histtype='bar',
                  X=X)
    return viz

def viz_wine(orientation="TD", max_depth=3, random_state=666, fancy=True, pickX=False):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    wine = load_wine()

    X_train = wine.data
    y_train = wine.target
    clf.fit(X_train, y_train)

    X = None
    if pickX:
        X = X_train[np.random.randint(0, len(X_train.data)),:]

    viz = dtreeviz(clf,
                   wine.data,
                   wine.target,
                   target_name='wine',
                   feature_names=wine.feature_names,
                   class_names=list(wine.target_names),
                   X=X)  # pass the test observation
    return viz


def weird_binary_case():
    # See bug https://github.com/parrt/dtreeviz/issues/17
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from dtreeviz.trees import dtreeviz

    x = np.random.choice([-1,1], size=(100, 2))
    y = np.random.choice([0, 1], size=100)

    viz = dtreeviz(
        tree_model=DecisionTreeClassifier(max_depth=1).fit(x, y),
        X_train=x,
        y_train=y,
        feature_names=['a', 'b'],
        target_name='y',
        class_names=[1, 0]
    )
    return viz


viz = weird_binary_case()
# viz = viz_wine(pickX=False, orientation='TD', max_depth=4, fancy=True)
# viz = viz_diabetes(pickX=True)
# viz = viz_boston(fancy=True, max_depth=4, orientation='TD')
# viz = viz_iris(fancy=True, orientation='TD', max_depth=3)
# viz = viz_digits(fancy=True, max_depth=3, orientation='TD')
# viz = viz_knowledge(fancy=True, orientation='TD', max_depth=2)
#g = graphviz.Source(st)

# tmp = tempfile.gettempdir()
# print(f"Tmp dir is {tmp}")
# with open("/tmp/t3.dot", "w") as f:
#     f.write(st+"\n")
#
#print(viz.dot)
# viz.save("/tmp/t.pdf")
# viz.save("/tmp/t.png")
# viz.save("/tmp/t.svg")
viz.view()
