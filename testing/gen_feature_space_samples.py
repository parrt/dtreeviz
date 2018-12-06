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
import inspect, sys, tempfile

from dtreeviz.trees import *

def viz_digits(features, feature_names, max_depth):
    digits = load_digits()

    # "8x8 image of integer pixels in the range 0..16."
    columns = [f'pixel[{i},{j}]' for i in range(8) for j in range(8)]

    fig, ax = plt.subplots(1, 1)
    X_train = digits.data[:,features]
    y_train = digits.target
    if len(features)==1:
        x_train = digits.data[:, features[0]]

        ctreeviz_univar(ax, x_train, y_train, max_depth=max_depth, feature_name=feature_names[0],
                        class_names=[str(i) for i in range(10)], gtype='strip', target_name='digit')
        filename = f"/tmp/digits-{feature_names[0]}-featspace-depth-{max_depth}.svg"
    else:
        ctreeviz_bivar(ax, X_train, y_train, max_depth=max_depth,
                       feature_names=feature_names, class_names=[str(i) for i in range(10)], target_name='digit')
        filename = f"/tmp/digits-{','.join(feature_names)}-featspace-depth-{max_depth}.svg"

    print(f"Create {filename}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=0, pad_inches=0)
    plt.show()
    # plt.close()

def viz_wine(features, feature_names, max_depth):
    wine = load_wine()

    X_train = wine.data[:,features]
    y_train = wine.target
    if len(features)==1:
        figsize = (6, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x_train = wine.data[:, features[0]]

        ctreeviz_univar(ax, x_train, y_train, max_depth=max_depth, feature_name=feature_names[0],
                        class_names=list(wine.target_names), gtype='strip', target_name='wine')
        filename = f"/tmp/wine-{feature_names[0]}-featspace-depth-{max_depth}.svg"
    else:
        figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ctreeviz_bivar(ax, X_train, y_train, max_depth=max_depth,
                       feature_names=feature_names, class_names=list(wine.target_names), target_name='wine',show={'splits'})
        filename = f"/tmp/wine-{','.join(feature_names)}-featspace-depth-{max_depth}.svg"

    print(f"Create {filename}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=0, pad_inches=0)
    plt.show()


def viz_knowledge(features, feature_names, max_depth):
    know = pd.read_csv("data/knowledge.csv")
    class_names = ['very_low', 'Low', 'Middle', 'High']
    know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(class_names)})

    X_train = know.drop('UNS', axis=1)
    X_train = X_train.values[:,features]
    y_train = know['UNS']
    if len(features)==1:
        figsize = (6, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x_train = know.PEG

        ctreeviz_univar(ax, x_train, y_train, max_depth=max_depth, feature_name=feature_names[0],
                        class_names=class_names, gtype='strip', target_name='knowledge')
        filename = f"/tmp/knowledge-{feature_names[0]}-featspace-depth-{max_depth}.svg"
    else:
        figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ctreeviz_bivar(ax, X_train, y_train, max_depth=max_depth,
                       feature_names=feature_names, class_names=class_names, target_name='knowledge')
        filename = f"/tmp/knowledge-{','.join(feature_names)}-featspace-depth-{max_depth}.svg"

    print(f"Create {filename}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=0, pad_inches=0)
    plt.show()


def viz_diabetes(features, feature_names, max_depth):
    diabetes = load_diabetes()

    X_train = diabetes.data
    X_train = X_train[:,features]
    y_train = diabetes.target
    if len(features)==1:
        figsize = (6, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x_train = diabetes.data[:, features[0]]

        rtreeviz_univar(ax, x_train, y_train, max_depth=max_depth, feature_name=feature_names[0], target_name='diabetes')
        filename = f"/tmp/diabetes-{feature_names[0]}-featspace-depth-{max_depth}.svg"
    else:
        figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        rtreeviz_bivar_heatmap(ax, X_train, y_train, max_depth=max_depth,
                               feature_names=feature_names)
        filename = f"/tmp/diabetes-{','.join(feature_names)}-featspace-depth-{max_depth}.svg"

    print(f"Create {filename}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=0, pad_inches=0)
    plt.show()


def viz_boston(features, feature_names, max_depth):
    boston = load_boston()

    X_train = boston.data
    X_train = X_train[:,features]
    y_train = boston.target
    if len(features)==1:
        figsize = (6, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x_train = boston.data[:, features[0]]

        rtreeviz_univar(ax, x_train, y_train, max_depth=max_depth, feature_name=feature_names[0], target_name='price')
        filename = f"/tmp/boston-{feature_names[0]}-featspace-depth-{max_depth}.svg"
    else:
        figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        rtreeviz_bivar_heatmap(ax, X_train, y_train, max_depth=max_depth,
                               feature_names=feature_names)
        filename = f"/tmp/boston-{','.join(feature_names)}-featspace-depth-{max_depth}.svg"

    print(f"Create {filename}")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=0, pad_inches=0)
    plt.show()


viz_boston(features=[5],feature_names=['RM'], max_depth=2)
viz_boston(features=[5],feature_names=['RM'], max_depth=4)
viz_boston(features=[5,12],feature_names=['RM','LSTAT'], max_depth=2)
viz_boston(features=[5,12],feature_names=['RM','LSTAT'], max_depth=4)

viz_diabetes(features=[2],feature_names=['bmi'], max_depth=2)
viz_diabetes(features=[2],feature_names=['bmi'], max_depth=5)
viz_diabetes(features=[2,0],feature_names=['bmi','age'], max_depth=2)
viz_diabetes(features=[2,0],feature_names=['bmi','age'], max_depth=5)

viz_knowledge(features=[4],feature_names=['PEG'], max_depth=2)
viz_knowledge(features=[4],feature_names=['PEG'], max_depth=3)
viz_knowledge(features=[4,3],feature_names=['PEG','LPR'], max_depth=2)
viz_knowledge(features=[4,3],feature_names=['PEG','LPR'], max_depth=3)

viz_wine(features=[12],feature_names=['proline'], max_depth=2)
viz_wine(features=[12],feature_names=['proline'], max_depth=3)
viz_wine(features=[12,6],feature_names=['proline','flavanoids'], max_depth=1)
viz_wine(features=[12,6],feature_names=['proline','flavanoids'], max_depth=2)
viz_wine(features=[12,6],feature_names=['proline','flavanoids'], max_depth=3)
viz_digits(features=[2*8+5], feature_names=['pixel[2,5]'], max_depth=20)
viz_digits(features=[4*8+4,2*8+5], feature_names=['pixel[4,4]','pixel[2,5]'], max_depth=5)