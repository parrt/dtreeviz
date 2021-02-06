import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from colour import Color
from PIL import ImageColor

from dtreeviz.colors import adjust_colors

def rfviz_bivar(model, X:np.ndarray, y:np.ndarray, ntiles=100, tile_fraction=.88,
                boundary_marker='o', boundary_markersize=.8,
                show_proba=True,
                colors=None, ax=None) -> None:
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    X1 = X[:, 0]
    X2 = X[:, 1]
    x1range = (min(X1), max(X1))
    x2range = (min(X2), max(X2))
    w = np.diff(np.linspace(*x1range, num=ntiles))[0]
    h = np.diff(np.linspace(*x2range, num=ntiles))[0]
    w *= tile_fraction
    h *= tile_fraction

    # Make sure we have one tile around border so instance circles don't overflow
    x1range = (min(X1) - w, max(X1) + w)
    x2range = (min(X2) - h, max(X2) + h)

    grid_points = []  # a list of coordinate pairs for the grid
    # Iterate through v1 (x-axis) most quickly then v2 (y-axis)
    for iv2, v2 in enumerate(np.linspace(*x2range, num=ntiles)):
        for iv1, v1 in enumerate(np.linspace(*x1range, num=ntiles)):
            grid_points.append([v1, v2])
    grid_points = np.array(grid_points)

    grid_proba = model.predict_proba(grid_points)
    grid_pred = np.argmax(grid_proba, axis=1)

    class_values = np.unique(y)
    class_X = [X[y == cl] for cl in class_values]

    rfviz_bivar_(grid_points, grid_proba, grid_pred, w, h, class_X, class_values,
                 ntiles=ntiles, boundary_marker=boundary_marker, boundary_markersize=boundary_markersize,
                 show_proba=show_proba,
                 colors=colors, ax=ax)


def rfviz_bivar_(grid_points, grid_proba, grid_pred, w, h, class_X, class_values,
                 ntiles=100, boundary_marker='o', boundary_markersize=.8,
                 show_proba=True,
                 colors=None, ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.8))

    nclasses = len(class_values)

    # Get class to color map
    colors = adjust_colors(colors)
    color_values = colors['classes'][nclasses]
    color_map = {v: color_values[i] for i, v in enumerate(class_values)}
    # multiply each probability vector times rgb color for each class then add
    # together to get weighted color
    rgb = np.array([ImageColor.getcolor(c, mode="RGB") for c in color_values])
    mycolors = grid_proba @ rgb
    mycolors /= 255  # get in [0..1]
    mycolors = [Color(rgb=c).hex for c in mycolors]
    y_pred_color = np.array(color_values)[grid_pred]

    # Draw probabilities or class prediction grid
    facecolors = mycolors if show_proba else y_pred_color
    boxes = []
    for i, (v1, v2) in enumerate(grid_points):
        # center a box over (v1,v2) grid location
        rect = patches.Rectangle((v1 - w / 2, v2 - h / 2), w, h, angle=0.0, linewidth=0,
                                 facecolor=facecolors[i], alpha=1.0)
        boxes.append(rect)
    # Adding collection is MUCH faster than repeated add_patch()
    ax.add_collection(PatchCollection(boxes, match_original=True))

    # Draw boundary locations
    # Get grid with class predictions with coordinates (x,y)
    # e.g., y_pred[0,0] is lower left pixel and y_pred[5,5] is top-right pixel
    # for npoints=5
    grid_pred = grid_pred.reshape(ntiles, ntiles)  # view as matrix
    dx = np.diff(grid_pred,
                 axis=1)  # find transitions from one class to the other moving horizontally
    dx = np.abs(dx)
    dx = np.hstack(
        [np.zeros((ntiles, 1)), dx])  # put a zero col vector on the left to restore size
    dy = np.diff(grid_pred, axis=0)  # find transitions moving vertically
    dy = np.abs(dy)
    dy = np.vstack(
        [np.zeros((1, ntiles)), dy])  # put a zero row vector on the top to restore size
    dx_edge_idx = np.where(
        dx.reshape(-1))  # what are the indexes of dx class transitions?
    dy_edge_idx = np.where(
        dy.reshape(-1))  # what are the indexes of dy class transitions?
    dx_edges = grid_points[
        dx_edge_idx]  # get v1,v2 coordinates of left-to-right transitions
    dy_edges = grid_points[
        dy_edge_idx]  # get v1,v2 coordinates of bottom-to-top transitions
    ax.plot(dx_edges[:, 0] - w / 2, dx_edges[:, 1], boundary_marker,
            markersize=boundary_markersize, c=colors['class_boundary'], alpha=1.0)
    ax.plot(dy_edges[:, 0], dy_edges[:, 1] - h / 2, boundary_marker,
            markersize=boundary_markersize, c=colors['class_boundary'], alpha=1.0)

    # Draw the X instances circles
    dot_w = 25
    for i, h in enumerate(class_X):
        ax.scatter(h[:, 0], h[:, 1], marker='o', s=dot_w, c=color_map[i],
                   edgecolors=colors['scatter_edge'], lw=.5, alpha=1.0)

    ax.spines['top'].set_visible(False)  # turns off the top "spine" completely
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)