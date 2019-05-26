# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import graphviz
import graphviz.backend
from numpy.distutils.system_info import f2py_info
from sklearn import tree
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, load_breast_cancer, load_diabetes, fetch_mldata
from matplotlib.figure import figaspect
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from dtreeviz.shadow import *
from numbers import Number
import matplotlib.patches as patches
from scipy import stats
from sklearn.neighbors import KernelDensity
import inspect
import sys
import tempfile

from dtreeviz.trees import *

"""
Generate samples into testing/samples dir to compare against future
images as a means of visually checking for errors.

Run with working directory as main dtreeviz dir so this code can see dtreeviz package
and data paths are set correctly. 
"""

# REGRESSION


def viz_boston(orientation="TD",
               max_depth=3,
               random_state=666,
               fancy=True,
               pickX=False,
               label_fontsize=12,
               ticks_fontsize=8,
               fontname="Arial"):
    regr = tree.DecisionTreeRegressor(
        max_depth=max_depth, random_state=random_state)
    boston = load_boston()

    regr.fit(boston.data, boston.target)

    X = None
    if pickX:
        X = boston.data[np.random.randint(0, len(boston.data)), :]

    if fontname == "TakaoPGothic":
        feature_names = list(map(lambda x: f"特徴量{x}", boston.feature_names))
    else:
        feature_names = boston.feature_names

    viz = dtreeviz(regr,
                   boston.data,
                   boston.target,
                   target_name='price',
                   feature_names=feature_names,
                   orientation=orientation,
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)

    return viz


def viz_diabetes(orientation="TD",
                 max_depth=3,
                 random_state=666,
                 fancy=True,
                 pickX=False,
                 label_fontsize=12,
                 ticks_fontsize=8,
                 fontname="Arial"):
    diabetes = load_diabetes()

    regr = tree.DecisionTreeRegressor(
        max_depth=max_depth, random_state=random_state)
    regr.fit(diabetes.data, diabetes.target)

    X = None
    if pickX:
        X = diabetes.data[np.random.randint(0, len(diabetes.data)), :]

    viz = dtreeviz(regr,
                   diabetes.data,
                   diabetes.target,
                   target_name='progr',
                   feature_names=diabetes.feature_names,
                   orientation=orientation,
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)

    return viz


def viz_sweets(orientation="TD",
               max_depth=3,
               random_state=666,
               fancy=True,
               pickX=False,
               label_fontsize=12,
               ticks_fontsize=8,
               fontname="Arial"):

    sweets = pd.read_csv("data/sweetrs.csv")
    sweets = sweets.sample(n=500)  # just grab 500 of 17k for plotting

    X_train, y_train = sweets.drop('rating', axis=1), sweets['rating']

    regr = tree.DecisionTreeRegressor(
        max_depth=max_depth, random_state=random_state)
    regr.fit(X_train, y_train)

    X = None
    if pickX:
        X = X_train.iloc[np.random.randint(0, len(X_train))]

    viz = dtreeviz(regr,
                   X_train,
                   y_train,
                   target_name='rating',
                   feature_names=sweets.columns,
                   orientation=orientation,
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)

    return viz


def viz_fires(orientation="TD",
              max_depth=3,
              random_state=666,
              fancy=True,
              pickX=False,
              label_fontsize=12,
              ticks_fontsize=8,
              fontname="Arial"):

    fires = pd.read_csv("data/forestfires.csv")
    fires['month'] = fires['month'].astype('category').cat.as_ordered()
    fires['month'] = fires['month'].cat.codes + 1
    fires['day'] = fires['day'].astype('category').cat.as_ordered()
    fires['day'] = fires['day'].cat.codes + 1

    X_train, y_train = fires.drop('area', axis=1), fires['area']

    regr = tree.DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=random_state)
    regr.fit(X_train, y_train)

    X = None
    if pickX:
        X = X_train.iloc[np.random.randint(0, len(X_train))].values

    viz = dtreeviz(regr,
                   X_train,
                   y_train,
                   target_name='area',
                   feature_names=fires.columns,
                   orientation=orientation,
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)

    return viz


# CLASSIFICATION

def viz_iris(orientation="TD",
             max_depth=3,
             random_state=666,
             fancy=True,
             pickX=False,
             label_fontsize=12,
             ticks_fontsize=8,
             fontname="Arial"):
    clf = tree.DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state)
    iris = load_iris()

    clf.fit(iris.data, iris.target)

    if fontname == "TakaoPGothic":
        feature_names = list(map(lambda x: f"特徴量{x}", iris.feature_names))
    else:
        feature_names = iris.feature_names

    X = None
    if pickX:
        X = iris.data[np.random.randint(0, len(iris.data)), :]

    viz = dtreeviz(clf,
                   iris.data,
                   iris.target,
                   target_name='variety',
                   feature_names=feature_names,
                   orientation=orientation,
                   class_names=["setosa",
                                "versicolor",
                                "virginica"],  # 0,1,2 targets
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)

    return viz


def viz_digits(orientation="TD",
               max_depth=3,
               random_state=666,
               fancy=True,
               pickX=False,
               label_fontsize=12,
               ticks_fontsize=8,
               fontname="Arial"):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=random_state)
    digits = load_digits()

    # "8x8 image of integer pixels in the range 0..16."
    columns = [f'pixel[{i},{j}]' for i in range(8) for j in range(8)]

    clf.fit(digits.data, digits.target)

    X = None
    if pickX:
        X = digits.data[np.random.randint(0, len(digits.data)), :]

    viz = dtreeviz(clf,
                   digits.data,
                   digits.target,
                   target_name='number',
                   feature_names=columns,
                   orientation=orientation,
                   class_names=[chr(c) for c in range(ord('0'), ord('9')+1)],
                   fancy=fancy,
                   histtype='bar',
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)
    return viz


def viz_wine(orientation="TD",
             max_depth=3,
             random_state=666,
             fancy=True,
             pickX=False,
             label_fontsize=12,
             ticks_fontsize=8,
             fontname="Arial"):
    clf = tree.DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state)
    wine = load_wine()

    clf.fit(wine.data, wine.target)

    X = None
    if pickX:
        X = wine.data[np.random.randint(0, len(wine.data)), :]

    viz = dtreeviz(clf,
                   wine.data,
                   wine.target,
                   target_name='wine',
                   feature_names=wine.feature_names,
                   orientation=orientation,
                   class_names=list(wine.target_names),
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)
    return viz


def viz_breast_cancer(orientation="TD",
                      max_depth=3,
                      random_state=666,
                      fancy=True,
                      pickX=False,
                      label_fontsize=12,
                      ticks_fontsize=8,
                      fontname="Arial"):
    clf = tree.DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state)
    cancer = load_breast_cancer()

    clf.fit(cancer.data, cancer.target)

    X = None
    if pickX:
        X = cancer.data[np.random.randint(0, len(cancer)), :]

    viz = dtreeviz(clf,
                   cancer.data,
                   cancer.target,
                   target_name='cancer',
                   feature_names=cancer.feature_names,
                   orientation=orientation,
                   class_names=list(cancer.target_names),
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)
    return viz


def viz_knowledge(orientation="TD",
                  max_depth=3,
                  random_state=666,
                  fancy=True,
                  pickX=False,
                  label_fontsize=12,
                  ticks_fontsize=8,
                  fontname="Arial"):
    # data from https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,
                                      random_state=random_state)
    know = pd.read_csv("data/knowledge.csv")
    target_names = ['very_low', 'Low', 'Middle', 'High']
    know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(target_names)})

    X_train, y_train = know.drop('UNS', axis=1), know['UNS']
    clf.fit(X_train, y_train)

    X = None
    if pickX:
        X = X_train.iloc[np.random.randint(0, len(know))]

    viz = dtreeviz(clf,
                   X_train,
                   y_train,
                   target_name='UNS',
                   feature_names=X_train.columns.values,
                   orientation=orientation,
                   class_names=target_names,
                   fancy=fancy,
                   X=X,
                   label_fontsize=label_fontsize,
                   ticks_fontsize=ticks_fontsize,
                   fontname=fontname)
    return viz


def save(name, dirname, orientation, max_depth, fancy=True, pickX=False, fontname="Arial"):
    print(f"Process {name} orientation={orientation} max_depth={max_depth} fancy={fancy}, pickX={pickX}, fontname={fontname}")

    viz = f(orientation=orientation,
            max_depth=max_depth,
            fancy=fancy,
            pickX=pickX,
            fontname=fontname)
    X = "-X" if pickX else ""
    filename = f"{name}-{orientation}-{max_depth}{X}-{fontname}"
    if not fancy:
        filename = filename+"-simple"
    print(f"{dirname}/{filename}.svg")
    viz.save(f"{dirname}/{filename}.svg")

    # do it the hard way to set dpi for png
    # g = graphviz.Source(st, format='png')
    # filepath = g.save(filename=f"{filename}.dot", directory=tempfile.gettempdir()) # save dot file
    # # cmd, rendered = graphviz.backend.command('dot', 'png', filepath)
    # cmd = ['dot', '-Gdpi=300', '-Tpng', f'-o{dirname}/{filename}.png', filepath]
    # graphviz.backend.run(cmd, capture_output=True, check=True, quiet=False)
    # That conversion fails to get good image. do this on command line:
    #
    # $ convert -density 300x300 boston-TD-2.pdf foo.png


if __name__ == '__main__':
    all_functions = inspect.getmembers(
        sys.modules[__name__], inspect.isfunction)
    these_functions = [t for t in all_functions if inspect.getmodule(
        t[1]) == sys.modules[__name__]]
    viz_funcs = [f[1] for f in these_functions if f[0].startswith('viz_')]

    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    else:
        dirname = "."

    print(f"tmp dir is {tempfile.gettempdir()}")
    for f in viz_funcs:
        name = f.__name__[len("viz_"):]
        #  if name!='sweets': continue
        save(name, dirname, "TD", 2)
        save(name, dirname, "TD", 4)
        if name == 'iris':
            save(name, dirname, "TD", 5)
            save(name, dirname, "TD", 5, pickX=True)
            save(name, dirname, "TD", 5, pickX=True, fontname="TakaoPGothic")
        if name == 'boston':
            #  save(name, dirname, "TD", 3)
            #  save(name, dirname, "TD", 5, fancy=False, pickX=True)
            #  save(name, dirname, "LR", 5, fancy=False, pickX=True)
            save(name, dirname, "LR", 5, fancy=False,
                 pickX=True, fontname="TakaoPGothic")
        if name == 'knowledge':
            save(name, dirname, "TD", 15, fancy=False, pickX=True)

        save(name, dirname, "LR", 3)
        save(name, dirname, "TD", 4, fancy=False)
        save(name, dirname, "LR", 2, pickX=True)
        save(name, dirname, "TD", 3, pickX=True)
