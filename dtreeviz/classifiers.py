from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageColor
from colour import Color
from matplotlib import patches as patches
from matplotlib.collections import PatchCollection

from dtreeviz import utils
from dtreeviz.colors import adjust_colors
from dtreeviz.utils import add_classifier_legend, _format_axes


def decision_boundaries(model, X: np.ndarray, y: np.ndarray,
                        ntiles=50, tile_fraction=.9,
                        binary_threshold=0.5,
                        show=['instances', 'boundaries', 'probabilities', 'misclassified', 'legend'],
                        feature_names=None, target_name=None, class_names=None,
                        markers=None,
                        boundary_marker='o', boundary_markersize=.8,
                        fontsize=9, fontname="Arial",
                        dot_w=25,
                        yshift=.08,
                        sigma=.013,
                        colors: dict = None,
                        ranges: Tuple = None,
                        figsize: Tuple = None,
                        ax=None) -> None:
    """
    Two-variable case:
    Draw a tiled grid over a 2D classifier feature space where each tile is colored by
    the coordinate probabilities or coordinate predicted class. The X,y instances
    are drawn on top of the tiling. The decision boundaries are indicated
    by dots in between the classes. You can specify a threshold for the binary
    classification case. Misclassified instances are highlighted.

    One-variable case:
    Draw a strip plot over a 1D feature space, one strip per class. A narrow rectangle
    along the bottom indicates a color combined probabilities from all classes. The
    color associated with the most likely class will dominate the probabilities rectangle.
    Misclassified instances are highlighted. Decision boundaries, where the predicted
    class shifts from one to another, are indicated by vertical dashed lines.

    TODO: assumes classes are contiguous and 0..k-1

    :param model: an sklearn or Keras classifier model or any other model that can answer
                  method predict_proba(X)
    :param X: A 1- or 2-column dataframe or numpy array with the one or two features to plot
    :param y: The target column with integers indicating the true instance classes;
              currently these must be contiguous 0..k-1 for k classes.
    :param ntiles: How many tiles to draw across the x1, x2 feature space
    :param tile_fraction: A value between 0..1 indicating how much of a tile
                          should be colored; e.g., .9 indicates the tile should leave
                          10% whitespace around the colored portion.
    :param boundary_marker: The marker symbol from matplotlib to use for the boundary;
                            default is a circle 'o'.
    :param boundary_markersize: The boundary marker size; default is .8
    :param feature_names: A list of strings indicating the one or two X variable names.
                          If None, no axes labels are showing
    :param target_name: If showing legend, this is the title of the legend box.
    :param class_names: If showing legend, these are the class names in the legend box
    :param show: Which elements to show, includes elements from
                 ['instances','boundaries','probabilities','misclassified','legend']
    :param markers: By default, just small circles are shown for each X instance, but
                    if not None, this is a list of matplotlib marker strings like ['X','s'].
    :param fontsize: Font size for tick labels and axis labels
    :param fontname: The font name for  tick labels and axis labels
    :param colors: A dictionary with adjustments to the colors
    :param dot_w: How wide should the circles be when drawing the instances
    :param yshift: For univariate case. If you'd like to play around with the strip plot,
                   this variable shifts the class clusters; a shifted zero puts them on
                   top of each other.
    :param sigma:  For univariate case. The standard deviation of the noise added to make
                   the strip plot.
    :param ranges: Tuple for ranges of plot. One range per input dimension also specified as tuple, 
                   e.g. ((10, 100), (500, 600)).
                   Ranges of plot are determined by min, max of X vector if not specified. 
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: An optional matplotlib "axes" upon which this method should draw. If you
               send in your own figure, it should be wide but not tall like shape 4,1
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if class_names is not None and np.max(y) >= len(class_names):
        raise ValueError(f"Target label values (for now) must be 0..{len(class_names)-1} for n={len(class_names)} labels")

    if model.__class__.__module__.startswith('tensorflow.python.keras') or \
            model.__class__.__module__.startswith('keras'):
        if not (hasattr(model, 'predict') and callable(getattr(model, 'predict'))):
            raise ValueError("Keras model argument must implement method `predict()`")
    elif not(hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))):
        raise ValueError("model argument must implement method `predict_proba()`")

    if len(X.shape) == 1 or (len(X.shape)==2 and X.shape[1] == 1):
        decision_boundaries_univar(model=model, x=X, y=y,
                                   ntiles=ntiles,
                                   binary_threshold=binary_threshold,
                                   show=show,
                                   feature_name=feature_names[0] if feature_names is not None else None,
                                   target_name=target_name,
                                   class_names=class_names,
                                   markers=markers,
                                   fontsize=fontsize, fontname=fontname,
                                   dot_w=dot_w,
                                   sigma=sigma,
                                   yshift=yshift,
                                   colors=colors,
                                   figsize=figsize,
                                   ax=ax)
    elif len(X.shape) == 2 and X.shape[1] == 2:
        decision_boundaries_bivar(model=model, X=X, y=y,
                                  ntiles=ntiles, tile_fraction=tile_fraction,
                                  binary_threshold=binary_threshold,
                                  show=show,
                                  feature_names=feature_names, target_name=target_name,
                                  class_names=class_names,
                                  markers=markers,
                                  boundary_marker=boundary_marker,
                                  boundary_markersize=boundary_markersize,
                                  fontsize=fontsize, fontname=fontname,
                                  dot_w=dot_w, colors=colors,
                                  ranges=ranges,
                                  figsize=figsize,
                                  ax=ax)
    else:
        raise ValueError(f"Expecting 2D data not {X.shape}")


def decision_boundaries_bivar(model, X:np.ndarray, y:np.ndarray,
                 ntiles=50, tile_fraction=.9,
                 binary_threshold=0.5,
                 show=['instances','boundaries','probabilities','misclassified','legend'],
                 feature_names=None, target_name=None, class_names=None,
                 markers=None,
                 boundary_marker='o', boundary_markersize=.8,
                 fontsize=9, fontname="Arial",
                 dot_w=25, colors:dict=None,
                 ranges=None,
                 figsize=None,
                 ax=None) -> None:
    """
    See comment and parameter descriptions for decision_boundaries() above.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if len(X.shape)==1 or (len(X.shape)==2 and X.shape[1]!=2) or len(X.shape)>2:
        raise ValueError(f"Expecting 2D data not {X.shape}")

    if ax is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

    # Created grid over the range of x1 and x2 variables, get probabilities, predictions
    grid_points, grid_proba, grid_pred_as_matrix, w, x_, class_X, class_values = \
        _compute_tiling(model, X, y, binary_threshold, ntiles, tile_fraction, ranges=ranges)

    x_proba = _predict_proba(model, X)
    if len(np.unique(y)) == 2:  # is k=2 binary?
        X_pred = np.where(x_proba[:, 1] >= binary_threshold, 1, 0)
    else:
        X_pred = np.argmax(x_proba, axis=1)  # TODO: assumes classes are 0..k-1
    class_X_pred = [X_pred[y == cl] for cl in class_values]

    if markers is None:
        markers = ['o']*len(class_X)

    colors = adjust_colors(colors)

    class_values = np.unique(y) # returns sorted

    # Get class to color map for probabilities and predictions
    color_map, grid_pred_colors, grid_proba_colors = \
        _get_grid_colors(grid_proba, grid_pred_as_matrix, class_values, colors)

    # Draw probabilities or class prediction grid
    facecolors = grid_proba_colors if 'probabilities' in show else grid_pred_colors
    _draw_tiles(ax, grid_points, facecolors, colors['tile_alpha'], x_, w)

    # Get grid with class predictions with coordinates (x,y)
    # e.g., y_pred[0,0] is lower left pixel and y_pred[5,5] is top-right pixel
    # for npoints=5
    grid_pred_as_matrix = grid_pred_as_matrix.reshape(ntiles, ntiles)

    if 'boundaries' in show:
        _draw_boundary_edges(ax, grid_points, grid_pred_as_matrix,
                             boundary_marker, boundary_markersize,
                             colors, w, x_)

    # Draw the X instances circles
    if 'instances' in show:
        for i, x_ in enumerate(class_X):
            if 'misclassified' in show:
                # Show correctly classified markers
                good_x = x_[class_X_pred[i] == class_values[i],:]
                ax.scatter(good_x[:, 0], good_x[:, 1],
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=colors['scatter_marker_alpha'],
                           edgecolors=colors['scatter_edge'],
                           lw=.5)
                # Show misclassified markers (can't have alpha per marker so do in 2 calls)
                bad_x = x_[class_X_pred[i] != class_values[i],:]
                ax.scatter(bad_x[:, 0], bad_x[:, 1],
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=1.0,
                           edgecolors=colors['warning'],
                           lw=.5)
            else:
                ax.scatter(x_[:, 0], x_[:, 1],
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=colors['scatter_marker_alpha'],
                           edgecolors=colors['scatter_edge'],
                           lw=.5)

    _format_axes(ax,
        feature_names[0] if feature_names is not None else None,
        feature_names[1] if feature_names is not None else None,
        colors, fontsize, fontname)

    if 'legend' in show:
        class_names = utils._normalize_class_names(class_names, nclasses=len(class_values))
        add_classifier_legend(ax, class_names, class_values, color_map, target_name, colors,
                              fontsize=fontsize, fontname=fontname)

def _compute_tiling(model, X:np.ndarray, y:np.ndarray, binary_threshold,
                    ntiles, tile_fraction, ranges):
    """
    Create grid over the range of x1 and x2 variables; use the model to
    compute the probabilities with model.predict_proba(), which will work with sklearn
    and, I think, XGBoost. Later we will have to figure out how to get probabilities
    out of the other models we support.

    The predictions are computed simply by picking the argmax of probabilities, which
    assumes classes are 0..k-1. TODO: update to allow disjoint integer class values

    For k=2 binary classifications, there is no way to set the threshold and so
    a threshold of 0.5 is implicitly chosen by argmax.

    This returns all of the details needed to plot the tiles. The coordinates of
    the grid are a linear space from min to max of each variable, inclusively.
    So if the range is 1..5 and we want 5 tiles, then the width of each tile is 1.
    We get a tile at each position. When we are drawing, the position is taken as
    the center of the tile. In this case, the grid points would be centered over
    1,2,3,4, and 5.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    X1 = X[:, 0]
    X2 = X[:, 1]

    if ranges is not None:
        x1range, x2range = ranges
        min_x1, max_x1 = x1range
        min_x2, max_x2 = x2range
    else:
        min_x1, max_x1 = min(X1), max(X1) 
        min_x2, max_x2 = min(X2), max(X2)
        
    x1r = max_x1 - min_x1
    x2r = max_x2 - min_x2
    border1 = x1r*0.05 # make a 5% border
    border2 = x2r*0.05
    x1range = (min_x1-border1, max_x1+border1)
    x2range = (min_x2-border2, max_x2+border2)
    w = (x1r+2*border1) / (ntiles-1)
    h = (x2r+2*border2) / (ntiles-1)
    w *= tile_fraction
    h *= tile_fraction

    grid_points = []  # a list of coordinate pairs for the grid
    # Iterate through v1 (x-axis) most quickly then v2 (y-axis)
    for iv2, v2 in enumerate(np.linspace(*x2range, num=ntiles, endpoint=True)):
        for iv1, v1 in enumerate(np.linspace(*x1range, num=ntiles, endpoint=True)):
            grid_points.append([v1, v2])
    grid_points = np.array(grid_points)

    class_values = np.unique(y)
    class_X = [X[y == cl] for cl in class_values]

    grid_proba = _predict_proba(model, grid_points)

    if len(np.unique(y))==2: # is k=2 binary?
        grid_pred = np.where(grid_proba[:,1]>=binary_threshold,1,0)
    else:
        grid_pred = np.argmax(grid_proba, axis=1) # TODO: assumes classes are 0..k-1

    return grid_points, grid_proba, grid_pred, w, h, class_X, class_values


def _get_grid_colors(grid_proba, grid_pred, class_values, colors):
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


def _draw_tiles(ax, grid_points, facecolors, tile_alpha, h, w):
    boxes = []
    for i, (v1, v2) in enumerate(grid_points):
        # center a box over (v1,v2) grid location
        rect = patches.Rectangle((v1 - w / 2, v2 - h / 2), w, h, angle=0.0, linewidth=0,
                                 facecolor=facecolors[i], alpha=tile_alpha)
        boxes.append(rect)
    # Adding collection is MUCH faster than repeated add_patch()
    ax.add_collection(PatchCollection(boxes, match_original=True))


def _draw_boundary_edges(ax, grid_points, grid_pred_as_matrix, boundary_marker, boundary_markersize,
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


def decision_boundaries_univar(model, x: np.ndarray, y: np.ndarray,
                               ntiles=100,
                               binary_threshold=0.5,
                               show=['instances', 'boundaries', 'probabilities', 'misclassified', 'legend'],
                               feature_name=None, target_name=None, class_names=None,
                               markers=None,
                               fontsize=9, fontname="Arial",
                               dot_w=25,
                               yshift=.09,
                               sigma=.09,
                               colors: dict = None,
                               figsize: Tuple = None,
                               ax=None) -> None:
    """
    See comment and parameter descriptions for decision_boundaries() above.
    """
    if ax is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    if (len(x.shape)==2 and x.shape[1]!=1) or len(x.shape)>2:
        raise ValueError(f"Expecting 1D data not {x.shape}")

    colors = adjust_colors(colors)

    mu = 0.08
    class_values = np.unique(y)
    nclasses = len(class_values)
    class_colors = np.array(colors['classes'][nclasses])
    color_map = {v: class_colors[i] for i, v in enumerate(class_values)}

    x1r = np.max(x) - np.min(x)
    x1range = (np.min(x), np.max(x))
    grid_points, w = np.linspace(*x1range, num=ntiles, endpoint=True, retstep=True)
    grid_proba = _predict_proba(model, grid_points)
    if len(np.unique(y)) == 2:  # is k=2 binary?
        grid_pred = np.where(grid_proba[:, 1] >= binary_threshold, 1, 0)
    else:
        grid_pred = np.argmax(grid_proba, axis=1)  # TODO: assumes classes are 0..k-1
    ymax = ax.get_ylim()[1]

    # compute the stripes on the bottom showing probabilities
    if 'probabilities' in show:
        class_values = np.unique(y)
        color_map, grid_pred_colors, grid_proba_colors = \
            _get_grid_colors(grid_proba, grid_pred, class_values, colors=adjust_colors(None))

        pred_box_height = .08 * ymax
        boxes = []
        for i, gx in enumerate(grid_points):
            rect = patches.Rectangle((gx, 0), w, pred_box_height,
                                     edgecolor='none', facecolor=grid_proba_colors[i],
                                     alpha=colors['tile_alpha'])
            boxes.append(rect)
        # drop box around the gradation
        ax.add_collection(PatchCollection(boxes, match_original=True))
        rect = patches.Rectangle((grid_points[0], 0), x1r + w, pred_box_height, linewidth=.3,
                                 edgecolor=colors['rect_edge'], facecolor='none')
        ax.add_patch(rect)

    if 'boundaries' in show:
        dx = np.abs(np.diff(grid_pred))
        dx = np.hstack([0, dx])
        dx_edge_idx = np.where(dx)  # indexes of dx class transitions?
        for lx in grid_points[dx_edge_idx]:
            ax.plot([lx, lx], [*ax.get_ylim()], '--', lw=.3,
                    c=colors['split_line'], alpha=1.0)

    if 'instances' in show:
        # user should pass in short and wide fig
        x_proba = _predict_proba(model, x)
        if len(np.unique(y)) == 2:  # is k=2 binary?
            x_pred = np.where(x_proba[:, 1] >= binary_threshold, 1, 0)
        else:
            x_pred = np.argmax(x_proba, axis=1)  # TODO: assumes classes are 0..k-1
        class_x = [x[y == cl] for cl in class_values]
        class_x_pred = [x_pred[y == cl] for cl in class_values]

        if markers is None:
            markers = ['o'] * len(class_x)
        for i, x_, in enumerate(class_x):
            if 'misclassified' in show:
                # Show correctly classified markers
                good_x = x_[class_x_pred[i] == class_values[i]]
                noise = np.random.normal(mu, sigma, size=len(good_x))
                ax.scatter(good_x, [mu + i * yshift] * len(good_x) + noise,
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=colors['scatter_marker_alpha'],
                           edgecolors=colors['scatter_edge'],
                           lw=.5)
                # Show misclassified markers (can't have alpha per marker so do in 2 calls)
                bad_x = x_[class_x_pred[i] != class_values[i]]
                noise = np.random.normal(mu, sigma, size=len(bad_x))
                ax.scatter(bad_x, [mu + i * yshift] * len(bad_x) + noise,
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=1.0,
                           edgecolors=colors['warning'],
                           lw=.5)
            else:
                noise = np.random.normal(mu, sigma, size=len(x_))
                ax.scatter(x_, [mu + i * yshift] * len(x_) + noise,
                           s=dot_w, c=color_map[i],
                           marker=markers[i],
                           alpha=colors['scatter_marker_alpha'],
                           edgecolors=colors['scatter_edge'],
                           lw=.5)

    _format_axes(ax, feature_name if feature_name is not None else None, None, colors, fontsize, fontname)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(0, mu + nclasses * yshift + 6*sigma)

    if 'legend' in show:
        class_names = utils._normalize_class_names(class_names, nclasses)
        add_classifier_legend(ax, class_names, class_values, color_map, target_name, colors,
                              fontsize=fontsize, fontname=fontname)


def _predict_proba(model, X):
    """
    This is where we figure out how to get a matrix of k probabilities for a k-class
    classification problem.  It works with any model that answers predict_proba()
    but we can add special cases such as Keras, that has deprecated that method.
    """
    if len(X.shape)==1:
        X = X.reshape(-1,1)
    # Keras wants predict not predict_proba and still gives probabilities
    if model.__class__.__module__.startswith('tensorflow.python.keras') or \
            model.__class__.__module__.startswith('keras'):
        proba = model.predict(X)
        if proba.shape[1]==1:
            proba = np.hstack([1-proba,proba]) # get prob y=0, y=1 nx2 matrix like sklearn
        return proba

    # sklearn etc...
    return model.predict_proba(X)
