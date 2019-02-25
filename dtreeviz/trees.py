from dtreeviz.utils import *

import numpy as np
import pandas as pd
import graphviz
from pathlib import Path
from sklearn import tree
from graphviz.backend import run, view
import matplotlib.pyplot as plt
from dtreeviz.shadow import *
from numbers import Number
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import tempfile
from os import getpid, makedirs
from sys import platform as PLATFORM
from colour import Color

YELLOW = "#fefecd" # "#fbfbd0" # "#FBFEB0"
GREEN = "#cfe2d4"
DARKBLUE = '#313695'
BLUE = '#4575b4'
DARKGREEN = '#006400'
LIGHTORANGE = '#fee090'
LIGHTBLUE = '#a6bddb'
GREY = '#444443'
WEDGE_COLOR = GREY #'orange'

HIGHLIGHT_COLOR = '#D67C03'

# How many bins should we have based upon number of classes
NUM_BINS = [0, 0, 10, 9, 8, 6, 6, 6, 5, 5, 5]
          # 0, 1, 2,  3, 4, 5, 6, 7, 8, 9, 10

color_blind_friendly_colors = [
    None, # 0 classes
    None, # 1 class
    ["#FEFEBB","#a1dab4"], # 2 classes
    ["#FEFEBB","#D9E6F5",'#a1dab4'], # 3 classes
    ["#FEFEBB","#D9E6F5",'#a1dab4','#fee090'], # 4
    ["#FEFEBB","#D9E6F5",'#a1dab4','#41b6c4','#fee090'], # 5
    ["#FEFEBB",'#c7e9b4','#41b6c4','#2c7fb8','#fee090','#f46d43'], # 6
    ["#FEFEBB",'#c7e9b4','#7fcdbb','#41b6c4','#225ea8','#fdae61','#f46d43'], # 7
    ["#FEFEBB",'#edf8b1','#c7e9b4','#7fcdbb','#1d91c0','#225ea8','#fdae61','#f46d43'], # 8
    ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43'], # 9
    ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43','#d73027'] # 10
]

class DTreeViz:
    def __init__(self,dot):
        self.dot = dot

    def _repr_svg_(self):
        return self.svg()

    def svg(self):
        """Render tree as svg and return svg text."""
        tmp = tempfile.gettempdir()
        svgfilename = f"{tmp}/DTreeViz_{getpid()}.svg"
        self.save(svgfilename)
        with open(svgfilename, encoding='UTF-8') as f:
            svg = f.read()
        return svg

    def view(self):
        tmp = tempfile.gettempdir()
        svgfilename = f"{tmp}/DTreeViz_{getpid()}.svg"
        self.save(svgfilename)
        view(svgfilename)

    def save(self, filename):
        """
        Save the svg of this tree visualization into filename argument.
        Mac platform can save any file type (.pdf, .png, .svg).  Other platforms
        would fail with errors. See https://github.com/parrt/dtreeviz/issues/4
        """
        path = Path(filename)
        if not path.parent.exists:
            makedirs(path.parent)

        g = graphviz.Source(self.dot, format='svg')
        dotfilename = g.save(directory=path.parent.as_posix(), filename=path.stem)

        if PLATFORM=='darwin':
            # dot seems broken in terms of fonts if we use -Tsvg. Force users to
            # brew install graphviz with librsvg (else metrics are off) and
            # use -Tsvg:cairo which fixes bug and also automatically embeds images
            format = path.suffix[1:]  # ".svg" -> "svg" etc...
            cmd = ["dot", f"-T{format}:cairo", "-o", filename, dotfilename]
            # print(' '.join(cmd))
            stdout, stderr = run(cmd, capture_output=True, check=True, quiet=False)

        else:
            if not filename.endswith(".svg"):
                raise (Exception(f"{PLATFORM} can only save .svg files: {filename}"))
            # Gen .svg file from .dot but output .svg has image refs to other files
            #orig_svgfilename = filename.replace('.svg', '-orig.svg')
            cmd = ["dot", "-Tsvg", "-o", filename, dotfilename]
            # print(' '.join(cmd))
            stdout, stderr = run(cmd, capture_output=True, check=True, quiet=False)

            # now merge in referenced SVG images to make all-in-one file
            with open(filename, encoding='UTF-8') as f:
                svg = f.read()
            svg = inline_svg_images(svg)
            with open(filename, "w", encoding='UTF-8') as f:
                f.write(svg)


def rtreeviz_univar(ax,
                    x_train: (pd.Series, np.ndarray),  # 1 vector of X data
                    y_train: (pd.Series, np.ndarray),
                    max_depth,
                    feature_name: str,
                    target_name: str,
                    fontsize: int = 14,
                    show={'title','splits'}):
    if isinstance(x_train, pd.Series):
        x_train = x_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    y_range = (min(y_train), max(y_train))  # same y axis for all
    overall_feature_range = (np.min(x_train), np.max(x_train))

    t = tree.DecisionTreeRegressor(max_depth=max_depth)
    t.fit(x_train.reshape(-1,1), y_train)

    shadow_tree = ShadowDecTree(t, x_train.reshape(-1,1), y_train, feature_names=[feature_name])
    splits = []
    for node in shadow_tree.internal:
        splits.append(node.split())
    splits = sorted(splits)
    bins = [overall_feature_range[0]] + splits + [overall_feature_range[1]]

    means = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        inrange = y_train[(x_train >= left) & (x_train < right)]
        means.append(np.mean(inrange))

    ax.scatter(x_train, y_train, marker='o', alpha=.4, c=BLUE,
                edgecolor=GREY, lw=.3)

    if 'splits' in show:
        for split in splits:
            ax.plot([split, split], [*y_range], '--', color='grey', linewidth=.7)

        prevX = overall_feature_range[0]
        for i, m in enumerate(means):
            split = overall_feature_range[1]
            if i < len(splits):
                split = splits[i]
            ax.plot([prevX, split], [m, m], '-', color='#f46d43', linewidth=2)
            prevX = split

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=fontsize)

    if 'title' in show:
        title = f"Regression tree depth {max_depth}, training $R^2$={t.score(x_train.reshape(-1,1),y_train):.3f}"
        plt.title(title, fontsize=fontsize, color=GREY)

    plt.xlabel(feature_name, fontsize=fontsize)
    plt.ylabel(target_name, fontsize=fontsize)


def rtreeviz_bivar_heatmap(ax, X_train, y_train, max_depth, feature_names,
                           fontsize=14, ticks_fontsize=12,
                           show={'title'}
                           ) -> tree.DecisionTreeClassifier:
    """
    Show tesselated 2D feature space for bivariate regression tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    if isinstance(X_train,pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    rt = tree.DecisionTreeRegressor(max_depth=max_depth)
    rt.fit(X_train, y_train)

    n_colors_in_map = 100
    y_lim = np.min(y_train), np.max(y_train)
    y_range = y_lim[1] - y_lim[0]
    color_map = list(str(c) for c in Color("#c7e9b4").range_to(Color("#081d58"), n_colors_in_map))

    shadow_tree = ShadowDecTree(rt, X_train, y_train, feature_names=feature_names)

    tesselation = shadow_tree.tesselation()

    for node,bbox in tesselation:
        pred = node.prediction()
        color = color_map[int(((pred - y_lim[0]) / y_range) * (n_colors_in_map-1))]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        rect = patches.Rectangle((x, y), w, h, 0, linewidth=.3, alpha=.5,
                                 edgecolor=GREY, facecolor=color)
        ax.add_patch(rect)

    colors = [color_map[int(((y-y_lim[0])/y_range)*(n_colors_in_map-1))] for y in y_train]
    x, y, z = X_train[:,0], X_train[:,1], y_train
    ax.scatter(x, y, marker='o', alpha=.95, c=colors, edgecolor=GREY, lw=.3)

    ax.set_xlabel(f"{feature_names[0]}", fontsize=fontsize, fontname="Arial", color=GREY)
    ax.set_ylabel(f"{feature_names[1]}", fontsize=fontsize, fontname="Arial", color=GREY)

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    if 'title' in show:
        accur = rt.score(X_train, y_train)
        title = f"Regression tree depth {max_depth}, training $R^2$={accur:.3f}"
        plt.title(title, fontsize=fontsize, color=GREY)

    return None


def rtreeviz_bivar_3D(ax, X_train, y_train, max_depth, feature_names, target_name,
                      fontsize=14, ticks_fontsize=10,
                      azim=0, elev=0, dist=7,
                      show={'title'}
                      ) -> tree.DecisionTreeClassifier:
    """
    Show 3D feature space for bivariate regression tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    if isinstance(X_train,pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    n_colors_in_map = 100

    ax.view_init(elev=elev, azim=azim)
    ax.dist=dist

    def plane(node, bbox):
        x = np.linspace(bbox[0], bbox[2], 2)
        y = np.linspace(bbox[1], bbox[3], 2)
        xx, yy = np.meshgrid(x, y)
        z = np.full(xx.shape, node.prediction())
        # print(f"{node.prediction()}->{int(((node.prediction()-y_lim[0])/y_range)*(n_colors_in_map-1))}, lim {y_lim}")
        # print(f"{color_map[int(((node.prediction()-y_lim[0])/y_range)*(n_colors_in_map-1))]}")
        ax.plot_surface(xx, yy, z, alpha=.85, shade=False,
                        color=color_map[int(((node.prediction()-y_lim[0])/y_range)*(n_colors_in_map-1))],
                        edgecolor=GREY, lw=.3)

    rt = tree.DecisionTreeRegressor(max_depth=max_depth)
    rt.fit(X_train, y_train)

    y_lim = np.min(y_train), np.max(y_train)
    y_range = y_lim[1] - y_lim[0]
    color_map = list(str(c) for c in Color("#c7e9b4").range_to(Color("#081d58"), n_colors_in_map))
    colors = [color_map[int(((y-y_lim[0])/y_range)*(n_colors_in_map-1))] for y in y_train]

    shadow_tree = ShadowDecTree(rt, X_train, y_train, feature_names=feature_names)
    tesselation = shadow_tree.tesselation()

    for node, bbox in tesselation:
        plane(node, bbox)

    x, y, z = X_train[:, 0], X_train[:, 1], y_train
    ax.scatter(x, y, z, marker='o', alpha=.7, edgecolor=GREY, lw=.3, c=colors)

    ax.set_xlabel(f"{feature_names[0]}", fontsize=fontsize, fontname="Arial", color=GREY)
    ax.set_ylabel(f"{feature_names[1]}", fontsize=fontsize, fontname="Arial", color=GREY)
    ax.set_zlabel(f"{target_name}", fontsize=fontsize, fontname="Arial", color=GREY)

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    if 'title' in show:
        accur = rt.score(X_train, y_train)
        title = f"Regression tree depth {max_depth}, training $R^2$={accur:.3f}"
        plt.title(title, fontsize=fontsize)

    return None


def ctreeviz_univar(ax, x_train, y_train, max_depth, feature_name, class_names,
                    target_name,
                    fontsize=14, nbins=25, gtype='strip',
                    show={'title','legend','splits'}):
    if isinstance(x_train, pd.Series):
        x_train = x_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    #    ax.set_facecolor('#F9F9F9')
    ct = tree.DecisionTreeClassifier(max_depth=max_depth)
    ct.fit(x_train.reshape(-1, 1), y_train)

    shadow_tree = ShadowDecTree(ct, x_train.reshape(-1, 1), y_train,
                                feature_names=[feature_name], class_names=class_names)

    n_classes = shadow_tree.nclasses()
    overall_feature_range = (np.min(x_train), np.max(x_train))
    class_values = shadow_tree.unique_target_values

    color_values = color_blind_friendly_colors[n_classes]
    colors = {v: color_values[i] for i, v in enumerate(class_values)}
    X_colors = [colors[cl] for cl in class_values]

    ax.set_xlabel(f"{feature_name}", fontsize=fontsize, fontname="Arial",
                  color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.3)

    r = overall_feature_range[1] - overall_feature_range[0]

    dot_w = 25

    X_hist = [x_train[y_train == cl] for cl in class_values]
    binwidth = r / nbins
    if gtype == 'barstacked':
        hist, bins, barcontainers = ax.hist(X_hist,
                                            color=X_colors,
                                            align='mid',
                                            histtype='barstacked',
                                            bins=np.arange(overall_feature_range[0],
                                                           overall_feature_range[
                                                               1] + binwidth, binwidth),
                                            label=class_names)

        for patch in barcontainers:
            for rect in patch.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(GREY)
        ax.set_xlim(*overall_feature_range)
        ax.set_xticks(overall_feature_range)
        ax.set_yticks([0, max([max(h) for h in hist])])
    elif gtype == 'strip':
        # user should pass in short and wide fig
        sigma = .013
        mu = .08
        class_step = .08
        dot_w = 20
        ax.set_ylim(0, mu + n_classes*class_step)
        for i, bucket in enumerate(X_hist):
            y_noise = np.random.normal(mu+i*class_step, sigma, size=len(bucket))
            ax.scatter(bucket, y_noise, alpha=.7, marker='o', s=dot_w, c=colors[i],
                       edgecolors=GREY, lw=.3)

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY,
                   labelsize=fontsize)

    splits = []
    for node in shadow_tree.internal:
        splits.append(node.split())
    splits = sorted(splits)
    bins = [ax.get_xlim()[0]] + splits + [ax.get_xlim()[1]]

    pred_box_height = .07 * ax.get_ylim()[1]
    preds = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        inrange = y_train[(x_train >= left) & (x_train < right)]
        values, counts = np.unique(inrange, return_counts=True)
        pred = values[np.argmax(counts)]
        rect = patches.Rectangle((left, 0), (right - left), pred_box_height, linewidth=.3,
                                 edgecolor=GREY, facecolor=colors[pred])
        ax.add_patch(rect)
        preds.append(pred)

    if 'legend' in show:
        add_classifier_legend(ax, class_names, class_values, colors, target_name)

    if 'title' in show:
        accur = ct.score(x_train.reshape(-1, 1), y_train)
        title = f"Classifier tree depth {max_depth}, training accuracy={accur*100:.2f}%"
        plt.title(title, fontsize=fontsize, color=GREY)

    if 'splits' in show:
        for split in splits:
            plt.plot([split, split], [*ax.get_ylim()], '--', color='grey', linewidth=1)


def ctreeviz_bivar(ax, X_train, y_train, max_depth, feature_names, class_names,
                   target_name,
                   fontsize=14,
                   show={'title','legend','splits'}):
    """
    Show tesselated 2D feature space for bivariate classification tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    if isinstance(X_train,pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    ct = tree.DecisionTreeClassifier(max_depth=max_depth)
    ct.fit(X_train, y_train)

    shadow_tree = ShadowDecTree(ct, X_train, y_train,
                                feature_names=feature_names, class_names=class_names)

    tesselation = shadow_tree.tesselation()

    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.unique_target_values

    color_values = color_blind_friendly_colors[n_classes]
    colors = {v: color_values[i] for i, v in enumerate(class_values)}

    if 'splits' in show:
        for node,bbox in tesselation:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]-bbox[0]
            h = bbox[3]-bbox[1]
            rect = patches.Rectangle((x, y), w, h, 0, linewidth=.3, alpha=.4,
                                     edgecolor=GREY, facecolor=colors[node.prediction()])
            ax.add_patch(rect)

    dot_w = 25
    X_hist = [X_train[y_train == cl] for cl in class_values]
    for i, h in enumerate(X_hist):
        ax.scatter(h[:,0], h[:,1], alpha=1, marker='o', s=dot_w, c=colors[i],
                   edgecolors=GREY, lw=.3)

    ax.set_xlabel(f"{feature_names[0]}", fontsize=fontsize, fontname="Arial", color=GREY)
    ax.set_ylabel(f"{feature_names[1]}", fontsize=fontsize, fontname="Arial", color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.3)

    if 'legend' in show:
        add_classifier_legend(ax, class_names, class_values, colors, target_name)

    if 'title' in show:
        accur = ct.score(X_train, y_train)
        title = f"Classifier tree depth {max_depth}, training accuracy={accur*100:.2f}%"
        plt.title(title, fontsize=fontsize, color=GREY)

    return None


def add_classifier_legend(ax, class_names, class_values, colors, target_name):
    # add boxes for legend
    boxes = []
    for i, c in enumerate(class_values):
        box = patches.Rectangle((0, 0), 20, 10, linewidth=.4, edgecolor=GREY,
                                facecolor=colors[c], label=class_names[c])
        boxes.append(box)
    leg = ax.legend(handles=boxes,
                    frameon=True,
                    shadow=False,
                    fancybox=True,
                    title=target_name,
                    handletextpad=.35,
                    borderpad=.8,
                    bbox_to_anchor=(1.0, 1.0),
                    edgecolor=GREY)
    leg.get_frame().set_linewidth(.5)
    leg.get_title().set_color(GREY)
    leg.get_title().set_fontsize(10)
    leg.get_title().set_fontweight('bold')
    for text in leg.get_texts():
        text.set_color(GREY)
        text.set_fontsize(10)


def dtreeviz(tree_model: (tree.DecisionTreeRegressor, tree.DecisionTreeClassifier),
             X_train: (pd.DataFrame, np.ndarray),
             y_train: (pd.Series, np.ndarray),
             feature_names: List[str],
             target_name: str,
             class_names: (Mapping[Number, str], List[str]) = None, # required if classifier
             precision: int = 2,
             orientation: ('TD', 'LR') = "TD",
             show_root_edge_labels: bool = True,
             show_node_labels: bool = False,
             fancy: bool = True,
             histtype: ('bar', 'barstacked', 'strip') = 'barstacked',
             highlight_path: List[int] = [],
             X: np.ndarray = None,
             max_X_features_LR: int = 10,
             max_X_features_TD: int = 20) \
    -> DTreeViz:
    """
    Given a decision tree regressor or classifier, create and return a tree visualization
    using the graphviz (DOT) language.

    :param tree_model: A DecisionTreeRegressor or DecisionTreeClassifier that has been
                       fit to X_train, y_train.
    :param X_train: A data frame or 2-D matrix of feature vectors used to train the model.
    :param y_train: A pandas Series or 1-D vector with target values or classes.
    :param feature_names: A list of the feature names.
    :param target_name: The name of the target variable.
    :param class_names: [For classifiers] A dictionary or list of strings mapping class
                        value to class name.
    :param precision: When displaying floating-point numbers, how many digits to display
                      after the decimal point. Default is 2.
    :param orientation:  Is the tree top down, "TD", or left to right, "LR"?
    :param show_root_edge_labels: Include < and >= on the edges emanating from the root?
    :param show_node_labels: Add "Node id" to top of each node in graph for educational purposes
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
    :type np.ndarray
    :param max_X_features_LR: If len(X) exceeds this limit for LR layout,
                            display only those features
                           used to guide X vector down tree. Helps when len(X) is large.
                           Default is 10.
    :param max_X_features_TD: If len(X) exceeds this limit for TD layout,
                            display only those features
                           used to guide X vector down tree. Helps when len(X) is large.
                           Default is 25.

    :return: A string in graphviz DOT language that describes the decision tree.
    """
    def node_name(node : ShadowDecTreeNode) -> str:
        return f"node{node.id}"

    def split_node(name, node_name, split):
        if fancy:
            labelgraph = node_label(node) if show_node_labels else ''
            html = f"""<table border="0">
            {labelgraph}
            <tr>
                    <td><img src="{tmp}/node{node.id}_{getpid()}.svg"/></td>
            </tr>
            </table>"""
        else:
            html = f"""<font face="Helvetica" color="#444443" point-size="12">{name}@{split}</font>"""
        if node.id in highlight_path:
            gr_node = f'{node_name} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
        else:
            gr_node = f'{node_name} [margin="0" shape=none label=<{html}>]'
        return gr_node


    def regr_leaf_node(node, label_fontsize: int = 12):
        # always generate fancy regr leaves for now but shrink a bit for nonfancy.
        labelgraph = node_label(node) if show_node_labels else ''
        html = f"""<table border="0">
        {labelgraph}
        <tr>
                <td><img src="{tmp}/leaf{node.id}_{getpid()}.svg"/></td>
        </tr>
        </table>"""
        if node.id in highlight_path:
            return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
        else:
            return f'leaf{node.id} [margin="0" shape=box penwidth="0" label=<{html}>]'


    def class_leaf_node(node, label_fontsize: int = 12):
        labelgraph = node_label(node) if show_node_labels else ''
        html = f"""<table border="0" CELLBORDER="0">
        {labelgraph}
        <tr>
                <td><img src="{tmp}/leaf{node.id}_{getpid()}.svg"/></td>
        </tr>
        </table>"""
        if node.id in highlight_path:
            return f'leaf{node.id} [margin="0" shape=box penwidth=".5" color="{HIGHLIGHT_COLOR}" style="dashed" label=<{html}>]'
        else:
            return f'leaf{node.id} [margin="0" shape=box penwidth="0" label=<{html}>]'

    def node_label(node):
        return f'<tr><td CELLPADDING="0" CELLSPACING="0"><font face="Helvetica" color="{GREY}" point-size="14"><i>Node {node.id}</i></font></td></tr>'

    def class_legend_html():
        return f"""
        <table border="0" cellspacing="0" cellpadding="0">
            <tr>
                <td border="0" cellspacing="0" cellpadding="0"><img src="{tmp}/legend_{getpid()}.svg"/></td>
            </tr>        
        </table>
        """

    def class_legend_gr():
        if not shadow_tree.isclassifier():
            return ""
        return f"""
            subgraph cluster_legend {{
                style=invis;
                legend [penwidth="0" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {class_legend_html()}
                >]
            }}
            """

    def instance_html(path, label_fontsize: int = 11):
        headers = []
        features_used = [node.feature() for node in path[:-1]] # don't include leaf
        display_X = X
        display_feature_names = feature_names
        highlight_feature_indexes = features_used
        if (orientation=='TD' and len(X)>max_X_features_TD) or\
           (orientation == 'LR' and len(X) > max_X_features_LR):
            # squash all features down to just those used
            display_X = [X[i] for i in features_used] + ['...']
            display_feature_names = [node.feature_name() for node in path[:-1]] + ['...']
            highlight_feature_indexes = range(0,len(features_used))

        for i,name in enumerate(display_feature_names):
            color = GREY
            if i in highlight_feature_indexes:
                color = HIGHLIGHT_COLOR
            headers.append(f'<td cellpadding="1" align="right" bgcolor="white"><font face="Helvetica" color="{color}" point-size="{label_fontsize}"><b>{name}</b></font></td>')

        values = []
        for i,v in enumerate(display_X):
            color = GREY
            if i in highlight_feature_indexes:
                color = HIGHLIGHT_COLOR
            if isinstance(v,int) or isinstance(v, str):
                disp_v = v
            else:
                disp_v = myround(v, precision)
            values.append(f'<td cellpadding="1" align="right" bgcolor="white"><font face="Helvetica" color="{color}" point-size="{label_fontsize}">{disp_v}</font></td>')

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
        pred, path = shadow_tree.predict(X)
        leaf = f"leaf{path[-1].id}"
        if shadow_tree.isclassifier():
            edge_label = f" Prediction<br/> {path[-1].prediction_name()}"
        else:
            edge_label = f" Prediction<br/> {myround(path[-1].prediction(), precision)}"
        return f"""
            subgraph cluster_instance {{
                style=invis;
                X_y [penwidth="0.3" margin="0" shape=box margin="0.03" width=.1, height=.1 label=<
                {instance_html(path)}
                >]
            }}
            {leaf} -> X_y [dir=back; penwidth="1.2" color="{HIGHLIGHT_COLOR}" label=<<font face="Helvetica" color="{GREY}" point-size="{11}">{edge_label}</font>>]
            """

    if orientation=="TD":
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
    # tmp = "/tmp"

    shadow_tree = ShadowDecTree(tree_model, X_train, y_train,
                                feature_names=feature_names, class_names=class_names)

    if X is not None:
        pred, path  = shadow_tree.predict(X)
        highlight_path = [n.id for n in path]

    n_classes = shadow_tree.nclasses()
    color_values = color_blind_friendly_colors[n_classes]

    # Fix the mapping from target value to color for entire tree
    colors = None
    if shadow_tree.isclassifier():
        class_values = shadow_tree.unique_target_values
        colors = {v:color_values[i] for i,v in enumerate(class_values)}

    y_range = (min(y_train)*1.03, max(y_train)*1.03) # same y axis for all

    if shadow_tree.isclassifier():
        # draw_legend_boxes(shadow_tree, f"{tmp}/legend")
        draw_legend(shadow_tree, target_name, f"{tmp}/legend_{getpid()}.svg")

    if isinstance(X_train,pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train,pd.Series):
        y_train = y_train.values

    # Find max height (count) for any bar in any node
    if shadow_tree.isclassifier():
        nbins = get_num_bins(histtype, n_classes)
        node_heights = shadow_tree.get_split_node_heights(X_train, y_train, nbins=nbins)

    internal = []
    for node in shadow_tree.internal:
        if fancy:
            if shadow_tree.isclassifier():
                class_split_viz(node, X_train, y_train,
                                filename=f"{tmp}/node{node.id}_{getpid()}.svg",
                                precision=precision,
                                colors=colors,
                                histtype=histtype,
                                node_heights=node_heights,
                                X = X,
                                highlight_node=node.id in highlight_path)
            else:
                regr_split_viz(node, X_train, y_train,
                               filename=f"{tmp}/node{node.id}_{getpid()}.svg",
                               target_name=target_name,
                               y_range=y_range,
                               precision=precision,
                               X=X,
                               highlight_node=node.id in highlight_path)

        nname = node_name(node)
        gr_node = split_node(node.feature_name(), nname, split=myround(node.split(), precision))
        internal.append(gr_node)

    leaves = []
    for node in shadow_tree.leaves:
        if shadow_tree.isclassifier():
            class_leaf_viz(node, colors=color_values,
                           filename=f"{tmp}/leaf{node.id}_{getpid()}.svg")
            leaves.append( class_leaf_node(node) )
        else:
            # for now, always gen leaf
            regr_leaf_viz(node, y_train, target_name=target_name,
                          filename=f"{tmp}/leaf{node.id}_{getpid()}.svg",
                          y_range=y_range, precision=precision)
            leaves.append( regr_leaf_node(node) )

    show_edge_labels = False
    all_llabel = '&lt;' if show_edge_labels else ''
    all_rlabel = '&ge;' if show_edge_labels else ''
    root_llabel = '&lt;' if show_root_edge_labels else ''
    root_rlabel = '&ge;' if show_root_edge_labels else ''

    edges = []
    # non leaf edges with > and <=
    for node in shadow_tree.internal:
        nname = node_name(node)
        if node.left.isleaf():
            left_node_name ='leaf%d' % node.left.id
        else:
            left_node_name = node_name(node.left)
        if node.right.isleaf():
            right_node_name ='leaf%d' % node.right.id
        else:
            right_node_name = node_name(node.right)
        llabel = all_llabel
        rlabel = all_rlabel
        if node==shadow_tree.root:
            llabel = root_llabel
            rlabel = root_rlabel
        lcolor = rcolor = GREY
        lpw = rpw = "0.3"
        if node.left.id in highlight_path:
            lcolor = HIGHLIGHT_COLOR
            lpw = "1.2"
        if node.right.id in highlight_path:
            rcolor = HIGHLIGHT_COLOR
            rpw = "1.2"
        edges.append( f'{nname} -> {left_node_name} [penwidth={lpw} color="{lcolor}" label=<{llabel}>]' )
        edges.append( f'{nname} -> {right_node_name} [penwidth={rpw} color="{rcolor}" label=<{rlabel}>]' )
        edges.append(f"""
        {{
            rank=same;
            {left_node_name} -> {right_node_name} [style=invis]
        }}
        """)

    newline = "\n\t"
    dot = f"""
digraph G {{
    splines=line;
    nodesep={nodesep};
    ranksep={ranksep};
    rankdir={orientation};
    margin=0.0;
    node [margin="0.03" penwidth="0.5" width=.1, height=.1];
    edge [arrowsize=.4 penwidth="0.3"]
    
    {newline.join(internal)}
    {newline.join(edges)}
    {newline.join(leaves)}
    
    {class_legend_gr()}
    {instance_gr()}
}}
    """

    return DTreeViz(dot)


def class_split_viz(node: ShadowDecTreeNode,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    colors: Mapping[int, str],
                    node_heights,
                    filename: str = None,
                    ticks_fontsize: int = 8,
                    label_fontsize: int = 9,
                    precision=1,
                    histtype: ('bar', 'barstacked', 'strip') = 'barstacked',
                    X : np.array = None,
                    highlight_node : bool = False
                    ):
    height_range = (.5, 1.5)
    h = prop_size(n=node_heights[node.id], counts=node_heights.values(), output_range=height_range)
    figsize=(3.3, h)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    feature_name = node.feature_name()

    # Get X, y data for all samples associated with this node.
    X_feature = X_train[:, node.feature()]
    X_feature, y_train = X_feature[node.samples()], y_train[node.samples()]

    n_classes = node.shadow_tree.nclasses()
    nbins = get_num_bins(histtype, n_classes)
    overall_feature_range = (np.min(X_train[:, node.feature()]), np.max(X_train[:, node.feature()]))

    overall_feature_range_wide = (overall_feature_range[0]-overall_feature_range[0]*.08,
                                  overall_feature_range[1]+overall_feature_range[1]*.05)

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial",
                  color=GREY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)

    class_names = node.shadow_tree.class_names

    r = overall_feature_range[1]-overall_feature_range[0]

    class_values = node.shadow_tree.unique_target_values
    X_hist = [X_feature[y_train == cl] for cl in class_values]

    if histtype=='strip':
        ax.yaxis.set_visible(False)
        ax.spines['left'].set_visible(False)
        sigma = .013
        mu = .05
        class_step = .08
        dot_w = 20
        ax.set_ylim(0, mu + n_classes * class_step)
        for i, bucket in enumerate(X_hist):
            alpha = .6 if len(bucket) > 10 else 1
            y_noise = np.random.normal(mu + i * class_step, sigma, size=len(bucket))
            ax.scatter(bucket, y_noise, alpha=alpha, marker='o', s=dot_w, c=colors[i],
                       edgecolors=GREY, lw=.3)
    else:
        X_colors = [colors[cl] for cl in class_values]
        binwidth = r / nbins

        hist, bins, barcontainers = ax.hist(X_hist,
                                            color=X_colors,
                                            align='mid',
                                            histtype=histtype,
                                            bins=np.arange(overall_feature_range[0],overall_feature_range[1] + binwidth, binwidth),
                                            label=class_names)
        # Alter appearance of each bar
        for patch in barcontainers:
            for rect in patch.patches:
                rect.set_linewidth(.5)
                rect.set_edgecolor(GREY)
        ax.set_yticks([0,max([max(h) for h in hist])])

    ax.set_xlim(*overall_feature_range_wide)
    ax.set_xticks(overall_feature_range)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    def wedge(ax,x,color):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xr = xmax - xmin
        yr = ymax - ymin
        hr = h / (height_range[1] - height_range[0])
        th = yr * .15 * 1 / hr  # convert to graph coordinates (ugh)
        tw = xr * .018
        tipy = -0.1 * yr * .15 * 1 / hr
        tria = np.array(
            [[x, tipy], [x - tw, -th], [x + tw, -th]])
        t = patches.Polygon(tria, facecolor=color)
        t.set_clip_on(False)
        ax.add_patch(t)
        ax.text(node.split(), -2 * th,
                f"{myround(node.split(),precision)}",
                horizontalalignment='center',
                fontsize=ticks_fontsize, color=GREY)

    wedge(ax, node.split(), color=WEDGE_COLOR)
    if highlight_node:
        wedge(ax, X[node.feature()], color=HIGHLIGHT_COLOR)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def class_leaf_viz(node : ShadowDecTreeNode,
                   colors : List[str],
                   filename: str):
    size = prop_size(node.nsamples(), counts=node.shadow_tree.leaf_sample_counts(),
                     output_range=(1.01, 1.5))
    # we visually need n=1 and n=9 to appear different but diff between 300 and 400 is no big deal
    size = np.sqrt(np.log(size))
    counts = node.class_counts()
    draw_piechart(counts, size=size, colors=colors, filename=filename, label=f"n={node.nsamples()}")


def regr_split_viz(node: ShadowDecTreeNode,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   target_name: str,
                   filename: str = None,
                   y_range=None,
                   ticks_fontsize: int = 8,
                   label_fontsize: int = 9,
                   precision=1,
                   X : np.array = None,
                   highlight_node : bool = False):
    figsize = (2.5, 1.1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=GREY)

    feature_name = node.feature_name()

    ax.set_xlabel(f"{feature_name}", fontsize=label_fontsize, fontname="Arial", color=GREY)

    ax.set_ylim(y_range)
    if node==node.shadow_tree.root:
        ax.set_ylabel(target_name, fontsize=label_fontsize, fontname="Arial", color=GREY)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    ax.tick_params(axis='both', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    # Get X, y data for all samples associated with this node.
    X_feature = X_train[:,node.feature()]
    X_feature, y_train = X_feature[node.samples()], y_train[node.samples()]

    overall_feature_range = (np.min(X_train[:,node.feature()]), np.max(X_train[:,node.feature()]))
    ax.set_xlim(*overall_feature_range)

    xmin, xmax = overall_feature_range
    xr = xmax - xmin

    xticks = list(overall_feature_range)
    if node.split()>xmin+.10*xr and node.split()<xmax-.1*xr: # don't show split if too close to axis ends
        xticks += [node.split()]
    ax.set_xticks(xticks)

    ax.scatter(X_feature, y_train, s=5, c=BLUE, alpha=.4, lw=.3)
    left, right = node.split_samples()
    left = y_train[left]
    right = y_train[right]
    split = node.split()
    ax.plot([overall_feature_range[0],split],[np.mean(left),np.mean(left)],'--', color='k', linewidth=1)
    ax.plot([split,split],[*y_range],'--', color='k', linewidth=1)
    ax.plot([split,overall_feature_range[1]],[np.mean(right),np.mean(right)],'--', color='k', linewidth=1)

    def wedge(ax,x,color):
        ymin, ymax = ax.get_ylim()
        xr = xmax - xmin
        yr = ymax - ymin
        hr = figsize[1]
        th = yr * .1
        tw = xr * .018
        tipy = ymin
        tria = np.array([[x, tipy], [x - tw, ymin-th], [x + tw, ymin-th]])
        t = patches.Polygon(tria, facecolor=color)
        t.set_clip_on(False)
        ax.add_patch(t)

    wedge(ax, node.split(), color=WEDGE_COLOR)

    if highlight_node:
        wedge(ax, X[node.feature()], color=HIGHLIGHT_COLOR)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def regr_leaf_viz(node : ShadowDecTreeNode,
                  y : (pd.Series,np.ndarray),
                  target_name,
                  filename:str=None,
                  y_range=None,
                  precision=1,
                  label_fontsize: int = 9,
                  ticks_fontsize: int = 8):
    samples = node.samples()
    y = y[samples]

    figsize = (.75, .8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.tick_params(colors=GREY)

    m = np.mean(y)

    ax.set_ylim(y_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.set_xticks([])
    # ax.set_yticks(y_range)

    ticklabelpad = plt.rcParams['xtick.major.pad']
    ax.annotate(f"{target_name}={myround(m,precision)}\nn={len(y)}",
                xy=(.5, 0), xytext=(.5, -.5*ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize = label_fontsize, fontname = "Arial", color = GREY)

    ax.tick_params(axis='y', which='major', width=.3, labelcolor=GREY, labelsize=ticks_fontsize)

    mu = .5
    sigma = .08
    X = np.random.normal(mu, sigma, size=len(y))
    ax.set_xlim(0, 1)
    alpha = .25

    ax.scatter(X, y, s=5, c='#225ea8', alpha=alpha, lw=.3)
    ax.plot([0,len(node.samples())],[m,m],'--', color=GREY, linewidth=1)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_legend(shadow_tree, target_name, filename):
    n_classes = shadow_tree.nclasses()
    class_values = shadow_tree.unique_target_values
    class_names = shadow_tree.class_names
    color_values = color_blind_friendly_colors[n_classes]
    colors = {v:color_values[i] for i,v in enumerate(class_values)}

    boxes = []
    for i, c in enumerate(class_values):
        box = patches.Rectangle((0, 0), 20, 10, linewidth=.4, edgecolor=GREY,
                                facecolor=colors[c], label=class_names[c])
        boxes.append(box)


    fig, ax = plt.subplots(1, 1, figsize=(1,1))
    leg = ax.legend(handles=boxes,
                    frameon=True,
                    shadow=False,
                    fancybox=True,
                    loc='center',
                    title=target_name,
                    handletextpad=.35,
                    borderpad=.8,
                    edgecolor=GREY)

    leg.get_frame().set_linewidth(.5)
    leg.get_title().set_color(GREY)
    leg.get_title().set_fontsize(10)
    leg.get_title().set_fontweight('bold')
    for text in leg.get_texts():
        text.set_color(GREY)
        text.set_fontsize(10)

    ax.set_xlim(0,20)
    ax.set_ylim(0,10)
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_piechart(counts,size,colors,filename,label=None):
    n_nonzero = np.count_nonzero(counts)
    i = np.nonzero(counts)[0][0]
    if n_nonzero==1:
        counts = [counts[i]]
        colors = [colors[i]]
    tweak = size * .01
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.axis('equal')
    # ax.set_xlim(0 - tweak, size + tweak)
    # ax.set_ylim(0 - tweak, size + tweak)
    ax.set_xlim(0, size-10*tweak)
    ax.set_ylim(0, size-10*tweak)
    # frame=True needed for some reason to fit pie properly (ugh)
    # had to tweak the crap out of this to get tight box around piechart :(
    wedges, _ = ax.pie(counts, center=(size/2-6*tweak,size/2-6*tweak), radius=size/2, colors=colors, shadow=False, frame=True)
    for w in wedges:
        w.set_linewidth(.5)
        w.set_edgecolor(GREY)

    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if label is not None:
        ax.text(size/2-6*tweak, -10*tweak, label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9, color=GREY, fontname="Arial")

    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def prop_size(n, counts, output_range = (0.00, 0.3)):
    min_samples = min(counts)
    max_samples = max(counts)
    sample_count_range = max_samples - min_samples


    if sample_count_range>0:
        zero_to_one = (n - min_samples) / sample_count_range
        return zero_to_one * (output_range[1] - output_range[0]) + output_range[0]
    else:
        return output_range[0]


def get_num_bins(histtype, n_classes):
    bins = NUM_BINS[n_classes]
    if histtype == 'barstacked':
        bins *= 2
    return bins


    global dot_already_tested
    if dot_already_tested: return
    dot_already_tested = True

    tmp = tempfile.gettempdir()
    dotfilename = f"{tmp}/testing_svg_{getpid()}.dot"
    with open(dotfilename, "w") as f:
        f.write("digraph G { A -> B }\n")
    svgfilename = f"{tmp}/testing_svg_{getpid()}.svg"
    cmd = ["dot", "-Tsvg:cairo", "-o", svgfilename, dotfilename]
    print(' '.join(cmd))
    ok = True
    try:
        os.execlp("dot", "dot", "-Tsvg:cairo", "-o", svgfilename, dotfilename)
        # run(cmd, capture_output=False, check=False, quiet=True)
    except:
        ok = False

    return ok
