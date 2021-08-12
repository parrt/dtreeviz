import os
import tempfile
from pathlib import Path
from sys import platform as PLATFORM
from typing import List

import numpy as np
import pandas as pd
import graphviz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from colour import Color, rgb2hex
from graphviz.backend import run, view
from sklearn import tree
from typing import Mapping, List, Tuple
from numbers import Number

from dtreeviz import interpretation as prediction_path
from dtreeviz.colors import adjust_colors
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.models.shadow_decision_tree import ShadowDecTreeNode
from dtreeviz.utils import inline_svg_images, myround, scale_SVG

# How many bins should we have based upon number of classes
NUM_BINS = [0, 0, 10, 9, 8, 6, 6, 6, 5, 5, 5]


# 0, 1, 2,  3, 4, 5, 6, 7, 8, 9, 10


class DTreeViz:
    def __init__(self, dot, scale=1.0):
        self.dot = dot
        self.scale = scale

    def _repr_svg_(self):
        return self.svg()

    def svg(self):
        """Render tree as svg and return svg text."""
        svgfilename = self.save_svg()
        with open(svgfilename, encoding='UTF-8') as f:
            svg = f.read()
        return svg

    def view(self):
        svgfilename = self.save_svg()
        view(svgfilename)

    def save_svg(self):
        """Saves the current object as SVG file in the tmp directory and returns the filename"""
        tmp = tempfile.gettempdir()
        svgfilename = os.path.join(tmp, f"DTreeViz_{os.getpid()}.svg")
        self.save(svgfilename)
        return svgfilename

    def save(self, filename):
        """
        Save the svg of this tree visualization into filename argument.
        Can only save .svg; others fail with errors.
        See https://github.com/parrt/dtreeviz/issues/4
        """
        path = Path(filename)
        if not path.parent.exists:
            os.makedirs(path.parent)

        g = graphviz.Source(self.dot, format='svg')
        dotfilename = g.save(directory=path.parent.as_posix(), filename=path.stem)
        format = path.suffix[1:]  # ".svg" -> "svg" etc...

        if not filename.endswith(".svg"):
            # Mac I think could do any format if we required:
            #   brew reinstall pango librsvg cairo
            raise (Exception(f"{PLATFORM} can only save .svg files: {filename}"))

        # Gen .svg file from .dot but output .svg has image refs to other files
        cmd = ["dot", f"-T{format}", "-o", filename, dotfilename]
        # print(' '.join(cmd))
        run(cmd, capture_output=True, check=True, quiet=False)

        if filename.endswith(".svg"):
            # now merge in referenced SVG images to make all-in-one file
            with open(filename, encoding='UTF-8') as f:
                svg = f.read()
            svg = inline_svg_images(svg)
            svg = scale_SVG(svg, self.scale)
            with open(filename, "w", encoding='UTF-8') as f:
                f.write(svg)


def rtreeviz_univar(tree_model,
                    x_data: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                    y_data: (pd.Series, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    tree_index: int = None,  # required in case of tree ensemble
                    ax=None,
                    fontsize: int = 14,
                    show={'title', 'splits'},
                    split_linewidth=.5,
                    mean_linewidth=2,
                    markersize=15,
                    colors=None):
    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, None,
                                                tree_index)
    x_data = shadow_tree.x_data.reshape(-1, )
    y_data = shadow_tree.y_data

    # ax as first arg is not good now that it's optional but left for compatibility reasons
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if x_data is None or y_data is None:
        raise ValueError(f"x_train and y_train must not be none")

    colors = adjust_colors(colors)

    y_range = (min(y_data), max(y_data))  # same y axis for all
    overall_feature_range = (np.min(x_data), np.max(x_data))

    splits = []
    for node in shadow_tree.internal:
        splits.append(node.split())
    splits = sorted(splits)
    bins = [overall_feature_range[0]] + splits + [overall_feature_range[1]]

    means = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        inrange = y_data[(x_data >= left) & (x_data <= right)]
        means.append(np.mean(inrange))

    ax.scatter(x_data, y_data, marker='o', alpha=colors['scatter_marker_alpha'], c=colors['scatter_marker'],
               s=markersize,
               edgecolor=colors['scatter_edge'], lw=.3)

    if 'splits' in show:
        for split in splits:
            ax.plot([split, split], [*y_range], '--', color=colors['split_line'], linewidth=split_linewidth)

        prevX = overall_feature_range[0]
        for i, m in enumerate(means):
            split = overall_feature_range[1]
            if i < len(splits):
                split = splits[i]
            ax.plot([prevX, split], [m, m], '-', color=colors['mean_line'], linewidth=mean_linewidth)
            prevX = split

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=fontsize)

    if 'title' in show:
        title = f"Regression tree depth {shadow_tree.get_max_depth()}, samples per leaf {shadow_tree.get_min_samples_leaf()},\nTraining $R^2$={shadow_tree.get_score()}"
        ax.set_title(title, fontsize=fontsize, color=colors['title'])

    ax.set_xlabel(shadow_tree.feature_names, fontsize=fontsize, color=colors['axis_label'])
    ax.set_ylabel(shadow_tree.target_name, fontsize=fontsize, color=colors['axis_label'])


def rtreeviz_bivar_heatmap(tree_model,
                           x_data: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                           y_data: (pd.Series, np.ndarray) = None,
                           feature_names: List[str] = None,
                           target_name: str = None,
                           tree_index: int = None,  # required in case of tree ensemble
                           ax=None,
                           fontsize=14, ticks_fontsize=12, fontname="Arial",
                           show={'title'},
                           n_colors_in_map=100,
                           colors=None,
                           markersize=15
                           ) -> tree.DecisionTreeClassifier:
    """
    Show tesselated 2D feature space for bivariate regression tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, None,
                                                tree_index)
    x_data = shadow_tree.x_data
    y_data = shadow_tree.y_data

    colors = adjust_colors(colors)

    y_lim = np.min(y_data), np.max(y_data)
    y_range = y_lim[1] - y_lim[0]
    color_map = [rgb2hex(c.rgb, force_long=True) for c in
                 Color(colors['color_map_min']).range_to(Color(colors['color_map_max']),
                                                         n_colors_in_map)]
    tesselation = shadow_tree.tesselation()

    for node, bbox in tesselation:
        pred = node.prediction()
        color = color_map[int(((pred - y_lim[0]) / y_range) * (n_colors_in_map - 1))]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle((x, y), w, h, 0, linewidth=.3, alpha=colors['tesselation_alpha'],
                                 edgecolor=colors['edge'], facecolor=color)
        ax.add_patch(rect)

    color_map = [color_map[int(((y - y_lim[0]) / y_range) * (n_colors_in_map - 1))] for y in y_data]
    x, y, z = x_data[:, 0], x_data[:, 1], y_data
    ax.scatter(x, y, marker='o', c=color_map, edgecolor=colors['scatter_edge'], lw=.3, s=markersize)

    ax.set_xlabel(f"{shadow_tree.feature_names[0]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.set_ylabel(f"{shadow_tree.feature_names[1]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=ticks_fontsize)

    if 'title' in show:
        accur = shadow_tree.get_score()
        title = f"Regression tree depth {shadow_tree.get_max_depth()}, training $R^2$={accur:.3f}"
        ax.set_title(title, fontsize=fontsize, color=colors['title'])

    return None


def rtreeviz_bivar_3D(tree_model,
                      x_data: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                      y_data: (pd.Series, np.ndarray) = None,
                      feature_names: List[str] = None,
                      target_name: str = None,
                      class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                      tree_index: int = None,  # required in case of tree ensemble
                      ax=None,
                      fontsize=14, ticks_fontsize=10, fontname="Arial",
                      azim=0, elev=0, dist=7,
                      show={'title'},
                      colors=None,
                      markersize=15,
                      n_colors_in_map=100
                      ) -> tree.DecisionTreeClassifier:
    """
    Show 3D feature space for bivariate regression tree. X_train should have
    just the 2 variables used for training.
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names,
                                                tree_index)
    x_data = shadow_tree.x_data
    y_data = shadow_tree.y_data

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    colors = adjust_colors(colors)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = dist

    def y_to_color_index(y):
        y_range = y_lim[1] - y_lim[0]
        return int(((y - y_lim[0]) / y_range) * (n_colors_in_map - 1))

    def plane(node, bbox, color_spectrum):
        x = np.linspace(bbox[0], bbox[2], 2)
        y = np.linspace(bbox[1], bbox[3], 2)
        xx, yy = np.meshgrid(x, y)
        z = np.full(xx.shape, node.prediction())
        ax.plot_surface(xx, yy, z, alpha=colors['tesselation_alpha_3D'], shade=False,
                        color=color_spectrum[y_to_color_index(node.prediction())],
                        edgecolor=colors['edge'], lw=.3)

    y_lim = np.min(y_data), np.max(y_data)
    color_spectrum = Color(colors['color_map_min']).range_to(Color(colors['color_map_max']), n_colors_in_map)
    color_spectrum = [rgb2hex(c.rgb, force_long=True) for c in color_spectrum]
    y_colors = [color_spectrum[y_to_color_index(y)] for y in y_data]
    tesselation = shadow_tree.tesselation()

    for node, bbox in tesselation:
        plane(node, bbox, color_spectrum)

    x, y, z = x_data[:, 0], x_data[:, 1], y_data
    ax.scatter(x, y, z, marker='o', alpha=colors['scatter_marker_alpha'], edgecolor=colors['scatter_edge'],
               lw=.3, c=y_colors, s=markersize)

    ax.set_xlabel(f"{shadow_tree.feature_names[0]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.set_ylabel(f"{shadow_tree.feature_names[1]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.set_zlabel(f"{shadow_tree.target_name}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=ticks_fontsize)

    if 'title' in show:
        accur = shadow_tree.get_score()
        title = f"Regression tree depth {shadow_tree.get_max_depth()}, training $R^2$={accur:.3f}"
        ax.set_title(title, fontsize=fontsize, color=colors['title'])

    return None


def ctreeviz_univar(tree_model,
                    x_data: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                    y_data: (pd.Series, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                    tree_index: int = None,  # required in case of tree ensemble
                    ax=None,
                    fontsize=14, fontname="Arial", nbins=25, gtype='strip',
                    show={'title', 'legend', 'splits'},
                    colors=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names,
                                                tree_index)
    x_data = shadow_tree.x_data.reshape(-1, )
    y_data = shadow_tree.y_data
    colors = adjust_colors(colors)
    n_classes = shadow_tree.nclasses()
    overall_feature_range = (np.min(x_data), np.max(x_data))
    class_values = shadow_tree.classes()
    color_values = colors['classes'][n_classes]
    color_map = {v: color_values[i] for i, v in enumerate(class_values)}
    X_colors = [color_map[cl] for cl in class_values]

    ax.set_xlabel(f"{shadow_tree.feature_names}", fontsize=fontsize, fontname=fontname,
                  color=colors['axis_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.3)

    X_hist = [x_data[y_data == cl] for cl in class_values]

    if gtype == 'barstacked':
        bins = np.linspace(start=overall_feature_range[0], stop=overall_feature_range[1], num=nbins, endpoint=True)
        hist, bins, barcontainers = ax.hist(X_hist,
                                            color=X_colors,
                                            align='mid',
                                            histtype='barstacked',
                                            bins=bins,
                                            label=shadow_tree.class_names)

        for patch in barcontainers:
            for rect in patch.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(colors['edge'])
        ax.set_xlim(*overall_feature_range)
        ax.set_xticks(overall_feature_range)
        ax.set_yticks([0, max([max(h) for h in hist])])
    elif gtype == 'strip':
        # user should pass in short and wide fig
        sigma = .013
        mu = .08
        class_step = .08
        dot_w = 20
        ax.set_ylim(0, mu + n_classes * class_step)
        for i, bucket in enumerate(X_hist):
            y_noise = np.random.normal(mu + i * class_step, sigma, size=len(bucket))
            ax.scatter(bucket, y_noise, alpha=colors['scatter_marker_alpha'], marker='o', s=dot_w, c=color_map[i],
                       edgecolors=colors['scatter_edge'], lw=.3)

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'],
                   labelsize=fontsize)

    splits = [node.split() for node in shadow_tree.internal]
    splits = sorted(splits)
    bins = [ax.get_xlim()[0]] + splits + [ax.get_xlim()[1]]

    if 'splits' in show:  # this gets the horiz bars showing prediction region
        pred_box_height = .07 * ax.get_ylim()[1]
        for i in range(len(bins) - 1):
            left = bins[i]
            right = bins[i + 1]
            inrange = y_data[(x_data >= left) & (x_data <= right)]
            values, counts = np.unique(inrange, return_counts=True)
            pred = values[np.argmax(counts)]
            rect = patches.Rectangle((left, 0), (right - left), pred_box_height, linewidth=.3,
                                     edgecolor=colors['edge'], facecolor=color_map[pred])
            ax.add_patch(rect)

    if 'legend' in show:
        add_classifier_legend(ax, shadow_tree.class_names, class_values, color_map, shadow_tree.target_name, colors, fontname=fontname)

    if 'title' in show:
        accur = shadow_tree.get_score()
        title = f"Classifier tree depth {shadow_tree.get_max_depth()}, training accuracy={accur * 100:.2f}%"
        ax.set_title(title, fontsize=fontsize, color=colors['title'])

    if 'splits' in show:
        for split in splits:
            ax.plot([split, split], [*ax.get_ylim()], '--', color=colors['split_line'], linewidth=1)


def ctreeviz_bivar(tree_model,
                   x_data: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                   y_data: (pd.Series, np.ndarray) = None,
                   feature_names: List[str] = None,
                   target_name: str = None,
                   class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                   tree_index: int = None,  # required in case of tree ensemble
                   ax=None,
                   fontsize=14,
                   fontname="Arial",
                   show={'title', 'legend', 'splits'},
                   colors=None):
    """
    Show tesselated 2D feature space for bivariate classification tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    # ax as first arg is not good now that it's optional but left for compatibility reasons
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names,
                                                tree_index)
    x_data = shadow_tree.x_data
    y_data = shadow_tree.y_data
    colors = adjust_colors(colors)
    tesselation = shadow_tree.tesselation()
    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.classes()
    color_values = colors['classes'][n_classes]
    color_map = {v: color_values[i] for i, v in enumerate(class_values)}

    if 'splits' in show:
        for node, bbox in tesselation:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle((x, y), w, h, 0, linewidth=.3, alpha=colors['tesselation_alpha'],
                                     edgecolor=colors['rect_edge'], facecolor=color_map[node.prediction()])
            ax.add_patch(rect)

    dot_w = 25
    X_hist = [x_data[y_data == cl] for cl in class_values]
    for i, h in enumerate(X_hist):
        ax.scatter(h[:, 0], h[:, 1], marker='o', s=dot_w, c=color_map[i],
                   edgecolors=colors['scatter_edge'], lw=.3)

    ax.set_xlabel(f"{shadow_tree.feature_names[0]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.set_ylabel(f"{shadow_tree.feature_names[1]}", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.3)

    if 'legend' in show:
        add_classifier_legend(ax, shadow_tree.class_names, class_values, color_map, shadow_tree.target_name, colors, fontname=fontname)

    if 'title' in show:
        accur = shadow_tree.get_score()
        title = f"Classifier tree depth {shadow_tree.get_max_depth()}, training accuracy={accur * 100:.2f}%"
        ax.set_title(title, fontsize=fontsize, color=colors['title'], )

    return None


def add_classifier_legend(ax, class_names, class_values, facecolors, target_name,
                          colors, fontsize=10, fontname='Arial'):
    # add boxes for legend
    boxes = []
    for c in class_values:
        box = patches.Rectangle((0, 0), 20, 10, linewidth=.4, edgecolor=colors['rect_edge'],
                                facecolor=facecolors[c], label=class_names[c])
        boxes.append(box)
    leg = ax.legend(handles=boxes,
                    frameon=True,
                    shadow=False,
                    fancybox=True,
                    handletextpad=.35,
                    borderpad=.8,
                    bbox_to_anchor=(1.0, 1.0),
                    edgecolor=colors['legend_edge'])

    leg.set_title(target_name, prop={'size': fontsize,
                                     'weight':'bold',
                                     'family':fontname})

    leg.get_frame().set_linewidth(.5)
    leg.get_title().set_color(colors['legend_title'])
    leg.get_title().set_fontsize(fontsize)
    leg.get_title().set_fontname(fontname)
    # leg.get_title().set_fontweight('bold')
    for text in leg.get_texts():
        text.set_color(colors['text'])
        text.set_fontsize(fontsize)


def dtreeviz(tree_model,
             x_data: (pd.DataFrame, np.ndarray) = None,
             y_data: (pd.DataFrame, np.ndarray) = None,
             feature_names: List[str] = None,
             target_name: str = None,
             class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
             tree_index: int = None,  # required in case of tree ensemble,
             precision: int = 2,
             orientation: ('TD', 'LR') = "TD",
             instance_orientation: ("TD", "LR") = "LR",
             show_root_edge_labels: bool = True,
             show_node_labels: bool = False,
             show_just_path: bool = False,
             fancy: bool = True,
             histtype: ('bar', 'barstacked', 'strip') = 'barstacked',
             highlight_path: List[int] = [],
             X: np.ndarray = None,
             max_X_features_LR: int = 10,
             max_X_features_TD: int = 20,
             depth_range_to_display: tuple = None,
             label_fontsize: int = 12,
             ticks_fontsize: int = 8,
             fontname: str = "Arial",
             title: str = None,
             title_fontsize: int = 14,
             colors: dict = None,
             scale=1.0
             ) \
        -> DTreeViz:
    """
    Given a decision tree regressor or classifier, create and return a tree visualization
    using the graphviz (DOT) language.

    We can call this function in two ways :
    1. by using shadow tree
        ex. dtreeviz(shadow_dtree)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target, [0, 1]))
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. dtreeviz(tree_model, dataset[features], dataset[target], features, target, class_names=[0, 1])
        - maintain backward compatibility

    :param tree_model: A DecisionTreeRegressor or DecisionTreeClassifier that has been
                       fit to X_train, y_data.
    :param X_train: A data frame or 2-D matrix of feature vectors used to train the model.
    :param y_data: A pandas Series or 1-D vector with target or classes values. These values should be numeric types.
    :param feature_names: A list of the feature names.
    :param target_name: The name of the target variable.
    :param class_names: [For classifiers] A dictionary or list of strings mapping class
                        value to class name.
    :param precision: When displaying floating-point numbers, how many digits to display
                      after the decimal point. Default is 2.
    :param orientation:  Is the tree top down, "TD", or left to right, "LR"?
    :param instance_orientation: table orientation (TD, LR) for showing feature prediction's values.
    :param show_root_edge_labels: Include < and >= on the edges emanating from the root?
    :param show_node_labels: Add "Node id" to top of each node in graph for educational purposes
    :param show_just_path: If True, it shows only the sample(X) prediction path
    :param fancy:
    :param histtype: [For classifiers] Either 'bar' or 'barstacked' to indicate
                     histogram type. We find that 'barstacked' looks great up to about.
                     four classes.
    :param highlight_path: A list of node IDs to highlight, default is [].
                           Useful for emphasizing node(s) in tree for discussion.
                           If X argument given then this is ignored.
    :type highlight_path: List[int]
    :param X: Instance to run down the tree; derived path to highlight from this vector.
              Show feature vector with labels underneath leaf reached. highlight_path
              is ignored if X is not None.
    :type X: np.ndarray
    :param label_fontsize: Size of the label font
    :param ticks_fontsize: Size of the tick font
    :param fontname: Font which is used for labels and text
    :param max_X_features_LR: If len(X) exceeds this limit for LR layout,
                            display only those features
                           used to guide X vector down tree. Helps when len(X) is large.
                           Default is 10.
    :param max_X_features_TD: If len(X) exceeds this limit for TD layout,
                            display only those features
                           used to guide X vector down tree. Helps when len(X) is large.
                           Default is 25.
    :param depth_range_to_display: range of depth levels to be displayed. The range values are inclusive
    :param title: An optional title placed at the top of the tree.
    :param title_fontsize: Size of the text for the title.
    :param scale: Default is 1.0. Scale the width, height of the overall SVG preserving aspect ratio
    :return: A string in graphviz DOT language that describes the decision tree.
    """

    def node_name(node: ShadowDecTreeNode) -> str:
        return f"node{node.id}"

    def split_node(name, node_name, split):
        if fancy:
            labelgraph = node_label(node) if show_node_labels else ''
            html = f"""<table border="0">
            {labelgraph}
            <tr>
                    <td><img src="{tmp}/node{node.id}_{os.getpid()}.svg"/></td>
            </tr>
            </table>"""
        else:
            html = f"""<font face="Helvetica" color="#444443" point-size="12">{name}@{split}</font>"""
        if node.id in highlight_path:
            gr_node = f'{node_name} [margin="0" shape=box penwidth=".5" color="{colors["highlight"]}" style="dashed" label=<{html}>]'
        else:
            gr_node = f'{node_name} [margin="0" shape=none label=<{html}>]'
        return gr_node

    def regr_leaf_node(node, label_fontsize: int = 12):
        # always generate fancy regr leaves for now but shrink a bit for nonfancy.
        labelgraph = node_label(node) if show_node_labels else ''
        html = f"""<table border="0">
        {labelgraph}
        <tr>
                <td><img src="{tmp}/leaf{node.id}_{os.getpid()}.svg"/></td>
        </tr>
        </table>"""
        if node.id in highlight_path:
            return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{colors["highlight"]}" style="dashed" label=<{html}>]'
        else:
            return f'leaf{node.id} [margin="0" shape=box penwidth="0" color="{colors["text"]}" label=<{html}>]'

    def class_leaf_node(node, label_fontsize: int = 12):
        labelgraph = node_label(node) if show_node_labels else ''
        html = f"""<table border="0" CELLBORDER="0">
        {labelgraph}
        <tr>
                <td><img src="{tmp}/leaf{node.id}_{os.getpid()}.svg"/></td>
        </tr>
        </table>"""
        if node.id in highlight_path:
            return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{colors["highlight"]}" style="dashed" label=<{html}>]'
        else:
            return f'leaf{node.id} [margin="0" shape=box penwidth="0" color="{colors["text"]}" label=<{html}>]'

    def node_label(node):
        return f'<tr><td CELLPADDING="0" CELLSPACING="0"><font face="Helvetica" color="{colors["node_label"]}" point-size="14"><i>Node {node.id}</i></font></td></tr>'

    def class_legend_html():
        return f"""
        <table border="0" cellspacing="0" cellpadding="0">
            <tr>
                <td border="0" cellspacing="0" cellpadding="0"><img src="{tmp}/legend_{os.getpid()}.svg"/></td>
            </tr>
        </table>
        """

    def class_legend_gr():
        if not shadow_tree.is_classifier():
            return ""
        return f"""
            subgraph cluster_legend {{
                style=invis;
                legend [penwidth="0" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {class_legend_html()}
                >]
            }}
            """

    def instance_html(path, instance_fontsize: int = 11):
        headers = []
        features_used = [node.feature() for node in path[:-1]]  # don't include leaf
        display_X = X
        display_feature_names = shadow_tree.feature_names
        highlight_feature_indexes = features_used
        if (orientation == 'TD' and len(X) > max_X_features_TD) or \
                (orientation == 'LR' and len(X) > max_X_features_LR):
            # squash all features down to just those used
            display_X = [X[i] for i in features_used] + ['...']
            display_feature_names = [node.feature_name() for node in path[:-1]] + ['...']
            highlight_feature_indexes = range(0, len(features_used))

        for i, name in enumerate(display_feature_names):
            if i in highlight_feature_indexes:
                color = colors['highlight']
            else:
                color = colors['text']
            headers.append(f'<td cellpadding="1" align="right" bgcolor="white">'
                           f'<font face="Helvetica" color="{color}" point-size="{instance_fontsize}">'
                           f'{name}'
                           '</font>'
                           '</td>')

        values = []
        for i, v in enumerate(display_X):
            if i in highlight_feature_indexes:
                color = colors['highlight']
            else:
                color = colors['text']
            if isinstance(v, int) or isinstance(v, str):
                disp_v = v
            else:
                disp_v = myround(v, precision)
            values.append(f'<td cellpadding="1" align="right" bgcolor="white">'
                          f'<font face="Helvetica" color="{color}" point-size="{instance_fontsize}">{disp_v}</font>'
                          '</td>')

        if instance_orientation == "TD":
            html_output = """<table border="0" cellspacing="0" cellpadding="0">"""
            for header, value in zip(headers, values):
                html_output += f"<tr> {header} {value} </tr>"
            html_output += "</table>"
            return html_output
        else:
            return f"""
                <table border="0" cellspacing="0" cellpadding="0">
                <tr>
                    {''.join(headers)}
                </tr>
                <tr>
                    {''.join(values)}
                </tr>
                </table>
                """

    def instance_gr():
        if X is None:
            return ""
        path = shadow_tree.predict_path(X)
        # print(f"path {[node.feature_name() for node in path]}")
        # print(f"path id {[node.id() for node in path]}")
        # print(f"path prediction {[node.prediction() for node in path]}")

        leaf = f"leaf{path[-1].id}"
        if shadow_tree.is_classifier():
            edge_label = f" &#160;Prediction<br/> {path[-1].prediction_name()}"
        else:
            edge_label = f" &#160;Prediction<br/> {myround(path[-1].prediction(), precision)}"
        return f"""
            subgraph cluster_instance {{
                style=invis;
                X_y [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {instance_html(path)}
                >]
            }}
            {leaf} -> X_y [dir=back; penwidth="1.2" color="{colors['highlight']}" label=<<font face="Helvetica" color="{colors['leaf_label']}" point-size="{11}">{edge_label}</font>>]
            """

    def get_internal_nodes():
        if show_just_path and X is not None:
            _internal = []
            for _node in shadow_tree.internal:
                if _node.id in highlight_path:
                    _internal.append(_node)
            return _internal
        else:
            return shadow_tree.internal

    def get_leaves():
        if show_just_path and X is not None:
            _leaves = []
            for _node in shadow_tree.leaves:
                if _node.id in highlight_path:
                    _leaves.append(_node)
                    break
            return _leaves
        else:
            return shadow_tree.leaves

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names,
                                                tree_index)
    colors = adjust_colors(colors)

    if orientation == "TD":
        ranksep = ".2"
        nodesep = "0.1"
    else:
        if fancy:
            ranksep = ".22"
            nodesep = "0.1"
        else:
            ranksep = ".05"
            nodesep = "0.09"

    tmp = tempfile.gettempdir()
    if X is not None:
        path = shadow_tree.predict_path(X)
        highlight_path = [n.id for n in path]

    n_classes = shadow_tree.nclasses()
    color_values = colors['classes'][n_classes]

    # Fix the mapping from target value to color for entire tree
    if shadow_tree.is_classifier():
        class_values = shadow_tree.classes()
        color_map = {v: color_values[i] for i, v in enumerate(class_values)}
        draw_legend(shadow_tree, shadow_tree.target_name, f"{tmp}/legend_{os.getpid()}.svg", colors=colors, fontname=fontname)

    X_data = shadow_tree.x_data
    y_data = shadow_tree.y_data
    if isinstance(X_data, pd.DataFrame):
        X_data = X_data.values
    if isinstance(y_data, pd.Series):
        y_data = y_data.values
    if y_data.dtype == np.dtype(object):
        try:
            y_data = y_data.astype('float')
        except ValueError as e:
            raise ValueError('y_data needs to consist only of numerical values. {}'.format(e))
        if len(y_data.shape) != 1:
            raise ValueError('y_data must a one-dimensional list or Pandas Series, got: {}'.format(y_data.shape))

    y_range = (min(y_data) * 1.03, max(y_data) * 1.03)  # same y axis for all

    # Find max height (count) for any bar in any node
    if shadow_tree.is_classifier():
        nbins = get_num_bins(histtype, n_classes)
        node_heights = shadow_tree.get_split_node_heights(X_data, y_data, nbins=nbins)

    internal = []
    for node in get_internal_nodes():
        if depth_range_to_display is not None:
            if node.level not in range(depth_range_to_display[0], depth_range_to_display[1] + 1):
                continue
        if fancy:
            if shadow_tree.is_classifier():
                class_split_viz(node, X_data, y_data,
                                filename=f"{tmp}/node{node.id}_{os.getpid()}.svg",
                                precision=precision,
                                colors={**color_map, **colors},
                                histtype=histtype,
                                node_heights=node_heights,
                                X=X,
                                ticks_fontsize=ticks_fontsize,
                                label_fontsize=label_fontsize,
                                fontname=fontname,
                                highlight_node=node.id in highlight_path)
            else:
                regr_split_viz(node, X_data, y_data,
                               filename=f"{tmp}/node{node.id}_{os.getpid()}.svg",
                               target_name=shadow_tree.target_name,
                               y_range=y_range,
                               precision=precision,
                               X=X,
                               ticks_fontsize=ticks_fontsize,
                               label_fontsize=label_fontsize,
                               fontname=fontname,
                               highlight_node=node.id in highlight_path,
                               colors=colors)

        nname = node_name(node)
        if not node.is_categorical_split():
            gr_node = split_node(node.feature_name(), nname, split=myround(node.split(), precision))
        else:
            gr_node = split_node(node.feature_name(), nname, split=node.split()[0])
        internal.append(gr_node)

    leaves = []
    for node in get_leaves():
        if depth_range_to_display is not None:
            if node.level not in range(depth_range_to_display[0], depth_range_to_display[1] + 1):
                continue
        if shadow_tree.is_classifier():
            class_leaf_viz(node, colors=color_values,
                           filename=f"{tmp}/leaf{node.id}_{os.getpid()}.svg",
                           graph_colors=colors,
                           fontname=fontname)
            leaves.append(class_leaf_node(node))
        else:
            # for now, always gen leaf
            regr_leaf_viz(node,
                          y_data,
                          target_name=shadow_tree.target_name,
                          filename=f"{tmp}/leaf{node.id}_{os.getpid()}.svg",
                          y_range=y_range,
                          precision=precision,
                          ticks_fontsize=ticks_fontsize,
                          label_fontsize=label_fontsize,
                          fontname=fontname,
                          colors=colors)
            leaves.append(regr_leaf_node(node))

    if show_just_path:
        show_root_edge_labels = False

    # TODO do we need show_edge_labels ?
    show_edge_labels = False
    all_llabel = '&lt;' if show_edge_labels else ''
    all_rlabel = '&ge;' if show_edge_labels else ''
    root_llabel = shadow_tree.get_root_edge_labels()[0] if show_root_edge_labels else ''
    root_rlabel = shadow_tree.get_root_edge_labels()[1] if show_root_edge_labels else ''

    edges = []
    # non leaf edges with > and <=
    for node in get_internal_nodes():
        if depth_range_to_display is not None:
            if node.level not in range(depth_range_to_display[0], depth_range_to_display[1]):
                continue
        nname = node_name(node)
        if node.left.isleaf():
            left_node_name = 'leaf%d' % node.left.id
        else:
            left_node_name = node_name(node.left)
        if node.right.isleaf():
            right_node_name = 'leaf%d' % node.right.id
        else:
            right_node_name = node_name(node.right)

        if node == shadow_tree.root:
            llabel = root_llabel
            rlabel = root_rlabel
        else:
            llabel = all_llabel
            rlabel = all_rlabel

        if node.is_categorical_split() and not shadow_tree.is_classifier():
            lcolor, rcolor = colors["categorical_split_left"], colors["categorical_split_right"]
        else:
            lcolor = rcolor = colors['arrow']

        lpw = rpw = "0.3"
        if node.left.id in highlight_path:
            lcolor = colors['highlight']
            lpw = "1.2"
        if node.right.id in highlight_path:
            rcolor = colors['highlight']
            rpw = "1.2"

        if show_just_path:
            if node.left.id in highlight_path:
                edges.append(f'{nname} -> {left_node_name} [penwidth={lpw} color="{lcolor}" label=<{llabel}>]')
            if node.right.id in highlight_path:
                edges.append(f'{nname} -> {right_node_name} [penwidth={rpw} color="{rcolor}" label=<{rlabel}>]')
        else:
            edges.append(f'{nname} -> {left_node_name} [penwidth={lpw} color="{lcolor}" label=<{llabel}>]')
            edges.append(f'{nname} -> {right_node_name} [penwidth={rpw} color="{rcolor}" label=<{rlabel}>]')
            edges.append(f"""
            {{
                rank=same;
                {left_node_name} -> {right_node_name} [style=invis]
            }}
            """)

    newline = "\n\t"
    if title:
        title_element = f'graph [label="{title}", labelloc=t, fontname="{fontname}" fontsize={title_fontsize} fontcolor="{colors["title"]}"];'
    else:
        title_element = ""
    dot = f"""
digraph G {{
    splines=line;
    nodesep={nodesep};
    ranksep={ranksep};
    rankdir={orientation};
    margin=0.0;
    {title_element}
    node [margin="0.03" penwidth="0.5" width=.1, height=.1];
    edge [arrowsize=.4 penwidth="0.3"]

    {newline.join(internal)}
    {newline.join(edges)}
    {newline.join(leaves)}

    {class_legend_gr()}
    {instance_gr()}
}}
    """

    return DTreeViz(dot, scale)


def class_split_viz(node: ShadowDecTreeNode,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    colors: dict,
                    node_heights,
                    filename: str = None,
                    ticks_fontsize: int = 8,
                    label_fontsize: int = 9,
                    fontname: str = "Arial",
                    precision=1,
                    histtype: ('bar', 'barstacked', 'strip') = 'barstacked',
                    X: np.array = None,
                    highlight_node: bool = False
                    ):

    def _get_bins(overall_range, nbins_):
        return np.linspace(start=overall_range[0], stop=overall_range[1], num=nbins_, endpoint=True)

    height_range = (.5, 1.5)
    h = prop_size(n=node_heights[node.id], counts=node_heights.values(), output_range=height_range)
    figsize = (3.3, h)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    feature_name = node.feature_name()

    # Get X, y data for all samples associated with this node.
    X_feature = X_train[:, node.feature()]
    X_node_feature, y_train = X_feature[node.samples()], y_train[node.samples()]

    n_classes = node.shadow_tree.nclasses()
    nbins = get_num_bins(histtype, n_classes)

    # for categorical splits, we need to ensure that each vertical bar will represent only one categorical feature value
    if node.is_categorical_split():
        feature_unique_size = np.unique(X_feature).size
        # keep the bar widths as uniform as possible for all node visualisations
        nbins = nbins if nbins > feature_unique_size else feature_unique_size + 1

    overall_feature_range = (np.min(X_train[:, node.feature()]), np.max(X_train[:, node.feature()]))

    overall_feature_range_wide = (overall_feature_range[0] - overall_feature_range[0] * .08,
                                  overall_feature_range[1] + overall_feature_range[1] * .05)

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)

    class_names = node.shadow_tree.class_names
    class_values = node.shadow_tree.classes()
    X_hist = [X_node_feature[y_train == cl] for cl in class_values]
    if histtype == 'strip':
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        sigma = .013
        mu = .05
        class_step = .08
        dot_w = 20
        ax.set_ylim(0, mu + n_classes * class_step)
        for i, bucket in enumerate(X_hist):
            alpha = colors['scatter_marker_alpha'] if len(bucket) > 10 else 1
            y_noise = np.random.normal(mu + i * class_step, sigma, size=len(bucket))
            ax.scatter(bucket, y_noise, alpha=alpha, marker='o', s=dot_w, c=colors[i],
                       edgecolors=colors['edge'], lw=.3)
    else:
        X_colors = [colors[cl] for cl in class_values]

        bins = _get_bins(overall_feature_range, nbins)
        hist, bins, barcontainers = ax.hist(X_hist,
                                            color=X_colors,
                                            align='mid',
                                            histtype=histtype,
                                            bins=bins,
                                            label=class_names)
        # Alter appearance of each bar
        for patch in barcontainers:
            for rect in patch.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(colors['rect_edge'])
        ax.set_yticks([0, max([max(h) for h in hist])])

    ax.set_xlim(*overall_feature_range_wide)
    ax.set_xticks(overall_feature_range)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=ticks_fontsize)

    def wedge(ax, x, color):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xr = xmax - xmin
        yr = ymax - ymin
        hr = h / (height_range[1] - height_range[0])
        th = yr * .15 * 1 / hr  # convert to graph coordinates (ugh)
        tw = xr * .018
        tipy = -0.1 * yr * .15 * 1 / hr

        if not node.is_categorical_split():
            tria = np.array(
                [[x, tipy], [x - tw, -th], [x + tw, -th]])
            t = patches.Polygon(tria, facecolor=color)
            t.set_clip_on(False)
            ax.add_patch(t)
            ax.text(node.split(), -2 * th,
                    f"{myround(node.split(), precision)}",
                    horizontalalignment='center',
                    fontsize=ticks_fontsize,
                    fontname=fontname,
                    color=colors['text_wedge'])
        else:
            left_split_values = node.split()[0]
            bins = _get_bins(overall_feature_range, nbins)
            for split_value in left_split_values:
                # to display the wedge exactly in the middle of the vertical bar
                for bin_index in range(len(bins) - 1):
                    if bins[bin_index] <= split_value <= bins[bin_index + 1]:
                        split_value = (bins[bin_index] + bins[bin_index + 1]) / 2
                        break
                tria = np.array(
                    [[split_value, tipy], [split_value - tw, -th], [split_value + tw, -th]])
                t = patches.Polygon(tria, facecolor=color)
                t.set_clip_on(False)
                ax.add_patch(t)

    wedge(ax, node.split(), color=colors['wedge'])
    if highlight_node:
        wedge(ax, X[node.feature()], color=colors['highlight'])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def class_leaf_viz(node: ShadowDecTreeNode,
                   colors: List[str],
                   filename: str,
                   graph_colors=None,
                   fontname: str = "Arial"):
    graph_colors = adjust_colors(graph_colors)
    # size = prop_size(node.nsamples(), counts=node.shadow_tree.leaf_sample_counts(),
    #                  output_range=(.2, 1.5))

    minsize = .15
    maxsize = 1.3
    slope = 0.02
    nsamples = node.nsamples()
    size = nsamples * slope + minsize
    size = min(size, maxsize)

    # we visually need n=1 and n=9 to appear different but diff between 300 and 400 is no big deal
    # size = np.sqrt(np.log(size))
    counts = node.class_counts()
    prediction = node.prediction_name()
    draw_piechart(counts, size=size, colors=colors, filename=filename, label=f"n={nsamples}\n{prediction}",
                  graph_colors=graph_colors, fontname=fontname)


def regr_split_viz(node: ShadowDecTreeNode,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   target_name: str,
                   filename: str = None,
                   y_range=None,
                   ticks_fontsize: int = 8,
                   label_fontsize: int = 9,
                   fontname: str = "Arial",
                   precision=1,
                   X: np.array = None,
                   highlight_node: bool = False,
                   colors: dict = None):
    colors = adjust_colors(colors)

    figsize = (2.5, 1.1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=colors['tick_label'])

    feature_name = node.feature_name()

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])

    ax.set_ylim(y_range)
    if node == node.shadow_tree.root:
        ax.set_ylabel(target_name, fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=ticks_fontsize)

    # Get X, y data for all samples associated with this node.
    X_feature = X_train[:, node.feature()]
    X_feature, y_train = X_feature[node.samples()], y_train[node.samples()]

    overall_feature_range = (np.min(X_train[:, node.feature()]), np.max(X_train[:, node.feature()]))
    ax.set_xlim(*overall_feature_range)
    xmin, xmax = overall_feature_range
    xr = xmax - xmin

    def wedge(ax, x, color):
        ymin, ymax = ax.get_ylim()
        xr = xmax - xmin
        yr = ymax - ymin
        th = yr * .1
        tw = xr * .018
        tipy = ymin
        tria = np.array([[x, tipy], [x - tw, ymin - th], [x + tw, ymin - th]])
        t = patches.Polygon(tria, facecolor=color)
        t.set_clip_on(False)
        ax.add_patch(t)

    if node.is_categorical_split() is False:

        xticks = list(overall_feature_range)
        if node.split() > xmin + .10 * xr and node.split() < xmax - .1 * xr: # don't show split if too close to axis ends
            xticks += [node.split()]
        ax.set_xticks(xticks)

        ax.scatter(X_feature, y_train, s=5, c=colors['scatter_marker'], alpha=colors['scatter_marker_alpha'], lw=.3)
        left, right = node.split_samples()
        left = y_train[left]
        right = y_train[right]
        split = node.split()

        # ax.scatter(X_feature, y_train, s=5, c=colors['scatter_marker'], alpha=colors['scatter_marker_alpha'], lw=.3)
        ax.plot([overall_feature_range[0], split], [np.mean(left), np.mean(left)], '--', color=colors['split_line'],
                linewidth=1)
        ax.plot([split, split], [*y_range], '--', color=colors['split_line'], linewidth=1)
        ax.plot([split, overall_feature_range[1]], [np.mean(right), np.mean(right)], '--', color=colors['split_line'],
                linewidth=1)
        wedge(ax, node.split(), color=colors['wedge'])

        if highlight_node:
            wedge(ax, X[node.feature()], color=colors['highlight'])
    else:
        left_index, right_index = node.split_samples()
        tw = (xmax - xmin) * .018

        ax.set_xlim(overall_feature_range[0] - tw, overall_feature_range[1] + tw)
        ax.scatter(X_feature[left_index], y_train[left_index], s=5, c=colors["categorical_split_left"], alpha=colors['scatter_marker_alpha'], lw=.3)
        ax.scatter(X_feature[right_index], y_train[right_index], s=5, c=colors["categorical_split_right"], alpha=colors['scatter_marker_alpha'], lw=.3)

        ax.plot([xmin - tw, xmax + tw], [np.mean(y_train[left_index]), np.mean(y_train[left_index])], '--', color=colors["categorical_split_left"],
                linewidth=1)
        ax.plot([xmin - tw, xmax + tw], [np.mean(y_train[right_index]), np.mean(y_train[right_index])], '--', color=colors["categorical_split_right"],
                linewidth=1)
        ax.set_xticks(np.unique(np.concatenate((X_feature, np.asarray(overall_feature_range)))))

        if highlight_node:
            wedge(ax, X[node.feature()], color=colors['highlight'])

    # plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def regr_leaf_viz(node: ShadowDecTreeNode,
                  y: (pd.Series, np.ndarray),
                  target_name,
                  filename: str = None,
                  y_range=None,
                  precision=1,
                  label_fontsize: int = 9,
                  ticks_fontsize: int = 8,
                  fontname: str = "Arial",
                  colors=None):
    colors = adjust_colors(colors)

    samples = node.samples()
    y = y[samples]

    figsize = (.75, .8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=colors['tick_label'])

    m = np.mean(y)

    ax.set_ylim(y_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.set_xticks([])
    # ax.set_yticks(y_range)

    ticklabelpad = plt.rcParams['xtick.major.pad']
    ax.annotate(f"{target_name}={myround(m, precision)}\nn={len(y)}",
                xy=(.5, 0), xytext=(.5, -.5 * ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])

    ax.tick_params(axis='y', which='major', width=.3, labelcolor=colors['tick_label'], labelsize=ticks_fontsize)

    mu = .5
    sigma = .08
    X = np.random.normal(mu, sigma, size=len(y))
    ax.set_xlim(0, 1)
    alpha = colors['scatter_marker_alpha']  # was .25

    ax.scatter(X, y, s=5, c=colors['scatter_marker'], alpha=alpha, lw=.3)
    ax.plot([0, len(node.samples())], [m, m], '--', color=colors['split_line'], linewidth=1)

    # plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_legend(shadow_tree, target_name, filename, colors=None, fontname="Arial"):
    colors = adjust_colors(colors)
    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.classes()
    class_names = shadow_tree.class_names
    color_values = colors['classes'][n_classes]
    color_map = {v: color_values[i] for i, v in enumerate(class_values)}

    boxes = []
    for i, c in enumerate(class_values):
        box = patches.Rectangle((0, 0), 20, 10, linewidth=.4, edgecolor=colors['rect_edge'],
                                facecolor=color_map[c], label=class_names[i])
        boxes.append(box)

    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    leg = ax.legend(handles=boxes,
                    frameon=True,
                    shadow=False,
                    fancybox=True,
                    loc='center',
                    title=target_name,
                    handletextpad=.35,
                    borderpad=.8,
                    edgecolor=colors['legend_edge'])

    leg.get_frame().set_linewidth(.5)
    leg.get_title().set_color(colors['legend_title'])
    leg.get_title().set_fontsize(10)
    leg.get_title().set_fontweight('bold')
    for text in leg.get_texts():
        text.set_color(colors['text'])
        text.set_fontsize(10)
        text.set_fontname(fontname)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_piechart(counts, size, colors, filename, label=None, fontname="Arial", graph_colors=None):
    graph_colors = adjust_colors(graph_colors)
    n_nonzero = np.count_nonzero(counts)

    if n_nonzero != 0:
        i = np.nonzero(counts)[0][0]
        if n_nonzero == 1:
            counts = [counts[i]]
            colors = [colors[i]]

    tweak = size * .01
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.axis('equal')
    # ax.set_xlim(0 - tweak, size + tweak)
    # ax.set_ylim(0 - tweak, size + tweak)
    ax.set_xlim(0, size - 10 * tweak)
    ax.set_ylim(0, size - 10 * tweak)
    # frame=True needed for some reason to fit pie properly (ugh)
    # had to tweak the crap out of this to get tight box around piechart :(
    wedges, _ = ax.pie(counts, center=(size / 2 - 6 * tweak, size / 2 - 6 * tweak), radius=size / 2, colors=colors,
                       shadow=False, frame=True)
    for w in wedges:
        w.set_linewidth(.5)
        w.set_edgecolor(graph_colors['pie'])

    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if label is not None:
        ax.text(size / 2 - 6 * tweak, -10 * tweak, label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9, color=graph_colors['text'], fontname=fontname)

    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def prop_size(n, counts, output_range=(0.00, 0.3)):
    min_samples = min(counts)
    max_samples = max(counts)
    sample_count_range = max_samples - min_samples

    if sample_count_range > 0:
        zero_to_one = (n - min_samples) / sample_count_range
        return zero_to_one * (output_range[1] - output_range[0]) + output_range[0]
    else:
        return output_range[0]


def get_num_bins(histtype, n_classes):
    bins = NUM_BINS[n_classes]
    if histtype == 'barstacked':
        bins *= 2
    return bins


def viz_leaf_samples(tree_model,
                     x_data: (pd.DataFrame, np.ndarray) = None,
                     feature_names: List[str] = None,
                     tree_index: int = None,  # required in case of tree ensemble
                     figsize: tuple = (10, 5),
                     display_type: str = "plot",
                     colors: dict = None,
                     fontsize: int = 14,
                     fontname: str = "Arial",
                     grid: bool = False,
                     bins: int = 10,
                     min_samples: int = 0,
                     max_samples: int = None):
    """Visualize the number of data samples from each leaf.

    Interpreting leaf samples can help us to see how the data is spread over the tree:
    - if we have a leaf with many samples and a good impurity, it means that we can be pretty confident
    on its prediction.
    - if we have a leaf with few samples and a good impurity, we cannot be very confident on its predicion and
    it could be a sign of overfitting.
    - by visualizing leaf samples, we can easily discover important leaves . Using describe_node_sample() function we
    can take all its samples and discover common patterns between leaf samples.
    - if the tree contains a lot of leaves and we want a general overview about leaves samples, we can use the
    parameter display_type='hist' to display the histogram of leaf samples.

    There is the option to filter the leaves with samples between 'min_samples' and 'max_samples'. This is helpful
    especially when you want to investigate leaves with number of samples from a specific range.


    We can call this function in two ways :
    1. by using shadow tree
        ex. viz_leaf_samples(shadow_dtree)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], features)
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. viz_leaf_samples(tree_model, dataset[features], dataset[target], features, target)
        - maintain backward compatibility

    TODO : put a link with notebook examples (at each function docs)

    This method contains three types of visualizations:
    - If display_type = 'plot' it will show leaf samples using a plot.
    - If display_type = 'text' it will show leaf samples as plain text. This method is preferred if number
    of leaves is very large and the plot become very big and hard to interpret.
    - If display_type = 'hist' it will show leaf sample histogram. Useful when you want to easily see the general
    distribution of leaf samples.

    Note : If the x_data and y_data are the datasets used to trained the model, then we will investigate the tree model
    as it was trained. We can give other x_data and y_data datasets, ex. validation dataset, to see how the new data is
    spread over the tree.

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param x_data: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :param figsize: tuple of int
        The plot size
    :param display_type: str, optional
       'plot', 'text'. 'hist'
    :param colors: dict
        The set of colors used for plotting
    :param fontsize: int
        Plot labels font size
    :param fontname: str
        Plot labels font name
    :param grid: bool
        True if we want to display the grid lines on the visualization
    :param bins: int
        Number of histogram bins
    :param min_samples: int
        Min number of samples for a leaf
    :param max_samples: int
        Max number of samples for a leaf
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, None, feature_names, None, None,
                                                tree_index)
    leaf_id, leaf_samples = shadow_tree.get_leaf_sample_counts(min_samples, max_samples)

    if display_type == "plot":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        ax.set_xticks(range(0, len(leaf_id)))
        ax.set_xticklabels(leaf_id)
        barcontainers = ax.bar(range(0, len(leaf_id)), leaf_samples, color=colors["hist_bar"], lw=.3, align='center',
                               width=1)
        for rect in barcontainers.patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel("leaf ids", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel("samples count", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(b=grid)
    elif display_type == "text":
        for leaf, samples in zip(leaf_id, leaf_samples):
            print(f"leaf {leaf} has {samples} samples")
    elif display_type == "hist":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        n, bins, patches = ax.hist(leaf_samples, bins=bins, color=colors["hist_bar"])
        for rect in patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel("leaf sample", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel("leaf count", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(b=grid)


def viz_leaf_criterion(tree_model,
                       tree_index: int = None,  # required in case of tree ensemble,
                       figsize: tuple = (10, 5),
                       display_type: str = "plot",
                       colors: dict = None,
                       fontsize: int = 14,
                       fontname: str = "Arial",
                       grid: bool = False,
                       bins: int = 10):
    """Visualize leaves criterion.

    The most common criterion/impurity for tree regressors is “mse”, “friedman_mse”, “mae” and for tree classifers are
    "gini" and "entropy". This information shows the leaf performance/confidence for its predictions, namely how pure or
    impure are the samples from each leaf. Each leaf performance, in the end, will determine the general tree performance.

    This visualisation can be used together with viz_leaf_samples() for a better leaf interpretation. For example,
    a leaf with good confidence, but few samples, can be a sign of overfitting. The best scenario would be to have a
    leaf with good confidence and also a lot of samples.

    We can call this function in two ways :
    1. by using shadow tree
        ex. viz_leaf_criterion(shadow_dtree)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target, [0, 1])
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. viz_leaf_criterion(tree_model)
        - maintain backward compatibility

    This method contains three types of visualizations:
    - a plot bar visualisations for each leaf criterion, when we want to interpret individual leaves
    - a hist visualizations with leaf criterion, when we want to have a general overview for all leaves
    - a text visualisations, useful when number of leaves is very large and visual interpretation becomes difficult.

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :param figsize: tuple of int
        The plot size
    :param display_type: str, optional
       'plot', 'text'. 'hist'
    :param colors: dict
        The set of colors used for plotting
    :param fontsize: int
        Plot labels font size
    :param fontname: str
        Plot labels font name
    :param grid: bool
        True if we want to display the grid lines on the visualization
    :param bins:  int
        Number of histogram bins
    :return:
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, None, None, None, None, None,
                                                tree_index)
    leaf_id, leaf_criteria = shadow_tree.get_leaf_criterion()

    if display_type == "plot":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        ax.set_xticks(range(0, len(leaf_id)))
        ax.set_xticklabels(leaf_id)
        barcontainers = ax.bar(range(0, len(leaf_id)), leaf_criteria, color=colors["hist_bar"], lw=.3, align='center',
                               width=1)
        for rect in barcontainers.patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel("leaf ids", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel(f"{shadow_tree.criterion()}", fontsize=fontsize, fontname=fontname,
                      color=colors['axis_label'])
        ax.grid(b=grid)
    elif display_type == "text":
        for leaf, criteria in zip(leaf_id, leaf_criteria):
            print(f"leaf {leaf} has {criteria} {shadow_tree.criterion()}")
    elif display_type == "hist":
        colors = adjust_colors(colors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        n, bins, patches = ax.hist(leaf_criteria, bins=bins, color=colors["hist_bar"])
        for rect in patches:
            rect.set_linewidth(.5)
            rect.set_edgecolor(colors['rect_edge'])
        ax.set_xlabel(f"{shadow_tree.criterion()}", fontsize=fontsize, fontname=fontname,
                      color=colors['axis_label'])
        ax.set_ylabel("leaf count", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(b=grid)


def ctreeviz_leaf_samples(tree_model,
                          x_data: (pd.DataFrame, np.ndarray) = None,
                          y_data: (pd.DataFrame, np.ndarray) = None,
                          feature_names: List[str] = None,
                          tree_index: int = None,  # required in case of tree ensemble,
                          figsize: tuple = (10, 5),
                          display_type: str = "plot",
                          plot_ylim: int = None,
                          colors: dict = None,
                          fontsize: int = 14,
                          fontname: str = "Arial",
                          grid: bool = False):
    """Visualize the number of data samples by class for each leaf.

    It's a good way to see how classes are distributed in leaves. For example, you can observe that in some
    leaves all the samples belong only to one class, or that in other leaves the distribution of classes is almost
    50/50.
    You could get all the samples from these leaves and look over/understand what they have in common. Now, you
    can understand your data in a model driven way.
    Right now it supports only binary classifications decision trees.

    We can call this function in two ways :
    1. by using shadow tree
        ex. ctreeviz_leaf_samples(shadow_dtree)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target, [0, 1])
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. ctreeviz_leaf_samples(tree_classifier, dataset[features], dataset[target], features)
        - maintain backward compatibility

    :param tree_model: tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param x_data: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param y_data: pd.Series, np.ndarray
        Target variable
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :param figsize: tuple of int, optional
        The plot size
    :param display_type: str, optional
       'plot' or 'text'
    :param plot_ylim: int, optional
        The max value for oY. This is useful in case we have few leaves with big sample values which 'shadow'
        the other leaves values.
    :param colors: dict
        The set of colors used for plotting
    :param fontsize: int
        Plot labels fontsize
    :param fontname: str
        Plot labels font name
    :param grid: bool
        True if we want to display the grid lines on the visualization
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, None, None,
                                                tree_index)
    index, leaf_samples_0, leaf_samples_1 = shadow_tree.get_leaf_sample_counts_by_class()

    if display_type == "plot":
        colors = adjust_colors(colors)
        colors_classes = colors['classes'][shadow_tree.nclasses()]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(.3)
        ax.spines['bottom'].set_linewidth(.3)
        ax.set_xticks(range(0, len(index)))
        ax.set_xticklabels(index)
        if plot_ylim is not None:
            ax.set_ylim(0, plot_ylim)

        bar_container0 = ax.bar(range(0, len(index)), leaf_samples_0, color=colors_classes[0], lw=.3, align='center',
                                width=1)
        bar_container1 = ax.bar(range(0, len(index)), leaf_samples_1, bottom=leaf_samples_0, color=colors_classes[1],
                                lw=.3, align='center', width=1)
        for bar_container in [bar_container0, bar_container1]:
            for rect in bar_container.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(colors['rect_edge'])

        ax.set_xlabel("leaf ids", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.set_ylabel("samples by class", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
        ax.grid(grid)
        ax.legend([bar_container0, bar_container1],
                  [f'class {shadow_tree.classes()[0]}', f'class {shadow_tree.classes()[1]}'])
    elif display_type == "text":
        for leaf, samples_0, samples_1 in zip(index, leaf_samples_0, leaf_samples_1):
            print(f"leaf {leaf}, samples : {samples_0}, {samples_1}")


def _get_leaf_target_input(shadow_tree: ShadowDecTree,
                           precision: int):
    x = []
    y = []
    means = []
    means_range = []
    x_labels = []
    sigma = .05
    for i, node in enumerate(shadow_tree.leaves):
        leaf_index_sample = node.samples()
        leaf_target = shadow_tree.y_data[leaf_index_sample]
        leaf_target_mean = np.mean(leaf_target)
        np.random.seed(0)  # generate the same list of random values for each call
        X = np.random.normal(i, sigma, size=len(leaf_target))

        x.extend(X)
        y.extend(leaf_target)
        means.append([leaf_target_mean, leaf_target_mean])
        means_range.append([i - (sigma * 3), i + (sigma * 3)])
        x_labels.append(f"{myround(leaf_target_mean, precision)}")

    return x, y, means, means_range, x_labels


def viz_leaf_target(tree_model,
                    x_data: (pd.DataFrame, np.ndarray) = None,
                    y_data: (pd.DataFrame, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    tree_index: int = None,  # required in case of tree ensemble,
                    show_leaf_labels: bool = True,
                    colors: dict = None,
                    markersize: int = 50,
                    label_fontsize: int = 14,
                    fontname: str = "Arial",
                    precision: int = 1,
                    figsize: tuple = None,
                    grid: bool = False,
                    prediction_line_width: int = 2):
    """Visualize leaf target distribution for regression decision trees.

    We can call this function in two ways :
    1. by using shadow tree
        ex. viz_leaf_target(shadow_dtree)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target)
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. viz_leaf_target(tree_model, dataset[features], dataset[target], features, target)
        - maintain backward compatibility


    :param tree_model: tree.DecisionTreeRegressor, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param x_data: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param y_data: pd.Series, np.ndarray
        Target variable values
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param target_name: str, optional
        The name of target variable
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :param show_leaf_labels: bool
        True if the plot should contains the leaf labels on x ax, False otherwise.
    :param markersize: int
        Marker size in points.
    :param precision: int
        When displaying floating-point numbers, how many digits to display after the decimal point. Default is 1.
    :param figsize: tuple
        Sets the (width, height) of the plot.
    :param grid: bool
        True if we want to display the grid lines on the visualization
    :param prediction_line_width: int
        The width of prediction line.
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, None,
                                                tree_index)
    x, y, means, means_range, y_labels = _get_leaf_target_input(shadow_tree, precision)
    colors = adjust_colors(colors)
    figsize = (np.log(len(y_labels)), np.log(len(y_labels)) * 1.5) if figsize is None else figsize
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.3)
    ax.spines['left'].set_linewidth(.3)

    ax.set_xlim(min(y), max(y) + 10)
    ax.set_ylim(-1, len(y_labels))
    ax.set_yticks(np.arange(0, len(y_labels), 1))
    ax.set_yticklabels([])
    ax.scatter(y, x, marker='o', alpha=colors['scatter_marker_alpha'] - 0.2, c=colors['scatter_marker'], s=markersize,
               edgecolor=colors['scatter_edge'], lw=.3)
    ax.set_xlabel(shadow_tree.target_name.lower(), fontsize=label_fontsize, fontname=fontname,
                  color=colors['axis_label'])
    ax.set_ylabel("leaf", fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])
    ax.grid(b=grid)

    if show_leaf_labels:
        for i in range(len(y_labels)):
            ax.text(max(y) + 10, i - 0.15, y_labels[i])
        ax.text(max(y) + 10, len(y_labels) - 0.15, shadow_tree.target_name.lower())

    for i in range(len(means)):
        ax.plot(means[i], means_range[i], color=colors['split_line'], linewidth=prediction_line_width)


def describe_node_sample(tree_model,
                         node_id: int,
                         x_data: (pd.DataFrame, np.ndarray) = None,
                         feature_names: List[str] = None,
                         tree_index: int = None,  # required in case of tree ensemble
                         ):
    """Generate stats (count, mean, std, etc) based on data samples from a specified node.

    This method is especially useful to investigate leaf samples from a decision tree. This is a way to discover data
    patterns, to better understand our tree model and to get new ideas for feature generation.

    We can call this function in two ways :
    1. by using shadow tree
        ex. describe_node_sample(shadow_dtree, node_id=10)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target)
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. describe_node_sample(tree_classifier, node_id=1, x_data=dataset[features], feature_names=features)
        - maintain backward compatibility

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param node_id: int
        Node id to interpret
    :param x_data: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :return: pd.DataFrame
        Node training samples' stats
    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, None, feature_names, None, None,
                                                tree_index)
    node_samples = shadow_tree.get_node_samples()
    return pd.DataFrame(shadow_tree.x_data, columns=shadow_tree.feature_names).iloc[node_samples[node_id]].describe()


def explain_prediction_path(tree_model,
                            x: np.ndarray,
                            x_data=None,
                            y_data=None,  # required for XGBoost
                            explanation_type: ('plain_english', 'sklearn_default') = "plain_english",
                            feature_names: List[str] = None,
                            target_name: str = None,
                            class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                            tree_index: int = None,  # required in case of tree ensemble
                            ):
    """Prediction path interpretation for a data instance.

    In case explanation_type = 'plain_english', there will be created a range of values for each feature, based on data
    instance values and its tree prediction path.
    A possible output for this method could be :
        1.5 <= Pclass
        3.5 <= Age < 44.5
        7.91 <= Fare < 54.25
        0.5 <= Sex_label
        Cabin_label < 3.5
        0.5 <= Embarked_label

    In case of explanation_type = 'sklean_default', where will be created a visualisation for feature importances,
    just like the popular one from sklearn library, but in this scencario, the feature importances will be calculated
    based only on the nodes from prediction path.


    We can call this function in two ways :
    1. by using shadow tree
        ex. explain_prediction_path(shadow_dtree, x)
        - we need to initialize shadow_tree before this call
            - ex. shadow_dtree = ShadowSKDTree(tree_model, dataset[features], dataset[target], features, target)
        - the main advantage is that we can use the shadow_tree for other visualisations methods as well
    2. by using sklearn, xgboost tree
        ex. explain_prediction_path(tree_model, x, feature_names=features, explanation_type="plain_english")
        - maintain backward compatibility

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param x: np.ndarray
        The data instance for which we want to investigate prediction path
    :param y_data: pd.Series, np.ndarray
        Target variable values
    :param explanation_type: plain_english, sklearn_default
        Specify the interpretation type
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param target_name: str, optional
        The name of target variable
    :param class_names: Mapping[Number, str], List[str], optional
        The list of class names. Required only for classifier
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.

    """

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, x_data, y_data, feature_names, None, class_names,
                                                tree_index)
    explainer = prediction_path.get_prediction_explainer(explanation_type)
    return explainer(shadow_tree, x)
