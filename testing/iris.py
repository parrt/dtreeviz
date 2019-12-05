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
                   fontname=fontname,
                   scale=(.5,.5))

    return viz

viz_iris().save("/tmp/t.svg")

# viz_iris().view()