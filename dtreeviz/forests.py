import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from colour import Color
from PIL import ImageColor

from dtreeviz.colors import adjust_colors, GREY
from dtreeviz.trees import ctreeviz_bivar, add_classifier_legend
from dtreeviz.models.shadow_decision_tree import ShadowDecTree


def ctreeviz_bivar_fusion(trees, X:np.ndarray, y:np.ndarray,
                          feature_names, target_name, class_names=None,
                          fontsize=12,
                          fontname="Arial",
                          show_region_edges=True,
                          alpha=.1,
                          ax=None):
    """
    Given a list of decision trees, overlap and fuse the feature space partitionings
    for two variables.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.spines['top'].set_visible(False)  # turns off the top "spine" completely
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    class_names = ShadowDecTree.normalize_class_names(class_names, nclasses=len(np.unique(y)))

    for i in range(len(trees)):
        ctreeviz_bivar(trees[i], X, y,
                       feature_names=feature_names, target_name=target_name,
                       class_names=class_names,
                       show={'splits'},
                       colors={'scatter_edge': 'black',
                               'tesselation_alpha': alpha,
                               'rect_edge':GREY if show_region_edges else None,
                               'scatter_marker_alpha':1.0},
                       fontsize=fontsize,
                       fontname=fontname,
                       ax=ax)


def crfviz_bivar(model, X:np.ndarray, y:np.ndarray, ntiles=50, tile_fraction=.9,
                 boundary_marker='o', boundary_markersize=.8,
                 show_proba=True,
                 feature_names=None, target_name=None, class_names=None,
                 show=['instances'],
                 fontsize=12,
                 fontname="Arial",
                 colors:dict=None, dot_w=25, ax=None) -> None:
    """
    (crfviz_bivar means "classifier random forest visualize, two variables")
    Draw a tiled grid over a 2D classifier feature space where each tile is colored by
    the coordinate probabilities or coordinate predicted class. The X,y instances
    are drawn as circles on top of the tiling. Draw dots representing the boundary
    between classes.

    Warning: there are a number of limitations in this initial implementation and
    so changes to the API or functionality are likely.
    :param model: an sklearn classifier model or any other model that can answer
                  method predict_proba(X)
    :param X: A 2-column data frame or numpy array with the two features to plot
    :param y: The target column with integers indicating the true instance classes;
              currently these must be contiguous 0..k-1 for k classes.
    :param ntiles: How many tiles to draw across the x1, x2 feature space
    :param tile_fraction: A value between 0..1 indicating how much of a tile
                          should be colored; e.g., .9 indicates the tile should leave
                          10% whitespace around the colored portion.
    :param boundary_marker: The marker symbol from matplotlib to use for the boundary;
                            default is a circle 'o'.
    :param boundary_markersize: The boundary marker size; default is .8
    :param show_proba: Show probabilities by default; if false, show prediction color.
    :param feature_names: 
    :param target_name: 
    :param class_names: 
    :param show: Which elements to show, includes elements from ['legend', 'instances'] 
    :param fontsize: 
    :param fontname: 
    :param colors: A dictionary with adjustments to the colors
    :param dot_w: How wide should the circles be when drawing the instances
    :param ax:  An optional matplotlib "axes" upon which this method should draw.
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.spines['top'].set_visible(False)  # turns off the top "spine" completely
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    # Created grid over the range of x1 and x2 variables, get probabilities, predictions
    grid_points, grid_proba, grid_pred_as_matrix, w, h, class_X, class_values = \
        compute_tiling(model, X, y, ntiles, tile_fraction)

    colors = adjust_colors(colors)

    # Get class to color map for probabilities and predictions
    color_map, grid_pred_colors, grid_proba_colors = \
        get_grid_colors(grid_proba, grid_pred_as_matrix, class_values, colors)

    # Draw probabilities or class prediction grid
    facecolors = grid_proba_colors if show_proba else grid_pred_colors
    draw_tiles(ax, grid_points, facecolors, h, w)

    # Get grid with class predictions with coordinates (x,y)
    # e.g., y_pred[0,0] is lower left pixel and y_pred[5,5] is top-right pixel
    # for npoints=5
    grid_pred_as_matrix = grid_pred_as_matrix.reshape(ntiles, ntiles)

    draw_boundary_edges(ax, grid_points, grid_pred_as_matrix,
                        boundary_marker, boundary_markersize,
                        colors, w, h)

    # Draw the X instances circles
    if 'instances' in show:
        for i, h in enumerate(class_X):
            ax.scatter(h[:, 0], h[:, 1], marker='o', s=dot_w, c=color_map[i],
                       edgecolors=colors['scatter_edge'], lw=.5, alpha=1.0)

    if feature_names is not None:
        ax.set_xlabel(f"{feature_names[0]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel(f"{feature_names[1]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])

    if 'legend' in show:
        class_names = ShadowDecTree.normalize_class_names(class_names,
                                                          nclasses=len(np.unique(y)))
        add_classifier_legend(ax, class_names, class_values, color_map, target_name, colors)


def compute_tiling(model, X:np.ndarray, y:np.ndarray, ntiles, tile_fraction):
    """
    Create grid over the range of x1 and x2 variables; use the model to
    compute the probabilities with model.predict_proba(), which will work with sklearn
    and, I think, XGBoost. Later we will have to figure out how to get probabilities
    out of the other models we support.

    The predictions are computed simply by picking the argmax of probabilities, which
    assumes classes are 0..k-1. TODO: update to allow this joint integer class values

    For k=2 binary classifications, there is no way to set the threshold and so
    a threshold of 0.5 his implicitly chosen by argmax.
    TODO: support threshold for binary case
    """
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

    class_values = np.unique(y)
    class_X = [X[y == cl] for cl in class_values]

    grid_proba = model.predict_proba(grid_points)
    grid_pred = np.argmax(grid_proba, axis=1) # TODO: assumes classes are 0..k-1

    return grid_points, grid_proba, grid_pred, w, h, class_X, class_values


def get_grid_colors(grid_proba, grid_pred, class_values, colors):
    """
    For the grid locations, return a list of colors, one per location
    indicating the class color.  To compute the probability color,
    we want to simulate overlaying regions from multiple trees onto
    the two-dimensional feature space using alpha to shade the colors.
    Instead, compute the color for each tile by combining the class colors
    according to their probabilities. If class 1 has probability .3 and class 2
    has probability .7, multiply the color ((R,G,B) color vector) associated
    with class 1 by .3 and the color vector associated with class 2 by .7 then
    add together. This gives a weighted color vector for each tile associated with
    the class probabilities. This gives the exact same effect as alpha channels,
    but transparent colors screwed up plotting the instance circles on top; they
    got washed out. This gives us more control and we can use alpha=1.
    """
    nclasses = len(class_values)
    class_colors = np.array(colors['classes'][nclasses])

    grid_pred_colors = class_colors[grid_pred] # color for each prediction in grid

    color_map = {v: class_colors[i] for i, v in enumerate(class_values)}
    # multiply each probability vector times rgb color for each class then add
    # together to get weighted color
    rgb = np.array([ImageColor.getcolor(c, mode="RGB") for c in class_colors])
    grid_proba_colors = grid_proba @ rgb
    grid_proba_colors /= 255  # get in [0..1]
    grid_proba_colors = [Color(rgb=c).hex for c in grid_proba_colors]
    return color_map, grid_pred_colors, grid_proba_colors


def draw_tiles(ax, grid_points, facecolors, h, w):
    boxes = []
    for i, (v1, v2) in enumerate(grid_points):
        # center a box over (v1,v2) grid location
        rect = patches.Rectangle((v1 - w / 2, v2 - h / 2), w, h, angle=0.0, linewidth=0,
                                 facecolor=facecolors[i], alpha=1.0)
        boxes.append(rect)
    # Adding collection is MUCH faster than repeated add_patch()
    ax.add_collection(PatchCollection(boxes, match_original=True))


def draw_boundary_edges(ax, grid_points, grid_pred_as_matrix, boundary_marker, boundary_markersize,
                        colors, w, h):
    ntiles = grid_pred_as_matrix.shape[0]

    # find transitions from one class to the other moving horizontally
    dx = np.diff(grid_pred_as_matrix, axis=1)
    dx = np.abs(dx)
    # put a zero col vector on the left to restore size
    dx = np.hstack([np.zeros((ntiles, 1)), dx])

    # find transitions moving vertically, bottom to top (grid matrix is flipped vertically btw)
    dy = np.diff(grid_pred_as_matrix, axis=0)
    dy = np.abs(dy)
    # put a zero row vector on the top to restore size
    dy = np.vstack([np.zeros((1, ntiles)), dy])

    dx_edge_idx = np.where(dx.reshape(-1)) # what are the indexes of dx class transitions?
    dy_edge_idx = np.where(dy.reshape(-1)) # what are the indexes of dy class transitions?
    dx_edges = grid_points[dx_edge_idx]    # get v1,v2 coordinates of left-to-right transitions
    dy_edges = grid_points[dy_edge_idx]    # get v1,v2 coordinates of bottom-to-top transitions

    # Plot the boundary markers in between tiles; e.g., shift dx stuff to the left half a tile
    ax.plot(dx_edges[:, 0] - w / 2, dx_edges[:, 1], boundary_marker,
            markersize=boundary_markersize, c=colors['class_boundary'], alpha=1.0)
    ax.plot(dy_edges[:, 0], dy_edges[:, 1] - h / 2, boundary_marker,
            markersize=boundary_markersize, c=colors['class_boundary'], alpha=1.0)