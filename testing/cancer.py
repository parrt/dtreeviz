# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import graphviz
import graphviz.backend
from numpy.distutils.system_info import f2py_info
from sklearn import tree
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata
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

viz_breast_cancer().view()