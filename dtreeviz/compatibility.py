# Functions to support backward compatibility to pre-2.0 API
import warnings
from numbers import Number
from typing import Mapping, List

import numpy as np
import pandas as pd
from sklearn import tree

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.utils import myround, DTreeVizRender
from dtreeviz.trees import DTreeVizAPI


def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = _warning_on_one_line


def rtreeviz_univar(tree_model,
                    X_train: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                    y_train: (pd.Series, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    tree_index: int = None,  # required in case of tree ensemble
                    ax=None,
                    fontsize: int = 10,
                    show={'title', 'splits'},
                    split_linewidth=.5,
                    mean_linewidth=2,
                    markersize=15,
                    colors=None):

    warnings.warn("rtreeviz_univar() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.rtree_feature_space(...)",
                  DeprecationWarning, stacklevel=2)

    if isinstance(feature_names, str):
        feature_names = [feature_names]
    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, None, tree_index)

    model = DTreeVizAPI(shadow_tree)
    model.rtree_feature_space(ax=ax, fontsize=fontsize, show=show, split_linewidth=split_linewidth,
                              mean_linewidth=mean_linewidth, markersize=markersize, colors=colors)


def rtreeviz_bivar_heatmap(tree_model,
                           X_train: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                           y_train: (pd.Series, np.ndarray) = None,
                           feature_names: List[str] = None,
                           target_name: str = None,
                           tree_index: int = None,  # required in case of tree ensemble
                           ax=None,
                           fontsize=10, ticks_fontsize=12, fontname="Arial",
                           show={'title'},
                           n_colors_in_map=100,
                           colors=None,
                           markersize=15
                           ) -> tree.DecisionTreeClassifier:
    """
    Show tesselated 2D feature space for bivariate regression tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    warnings.warn("rtreeviz_bivar_heatmap() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.rtree_feature_space(...)",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, None,
                                                tree_index)

    model = DTreeVizAPI(shadow_tree)
    model.rtree_feature_space(ax=ax, fontsize=fontsize, ticks_fontsize=ticks_fontsize, fontname=fontname, show=show,
                              n_colors_in_map=n_colors_in_map, colors=colors, markersize=markersize)

def rtreeviz_bivar_3D(tree_model,
                      X_train: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                      y_train: (pd.Series, np.ndarray) = None,
                      feature_names: List[str] = None,
                      target_name: str = None,
                      class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                      tree_index: int = None,  # required in case of tree ensemble
                      ax=None,
                      fontsize=10, ticks_fontsize=10, fontname="Arial",
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
    warnings.warn("rtreeviz_bivar_3D() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.rtree_feature_space3D(...)",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names,
                                                tree_index)

    model = DTreeVizAPI(shadow_tree)
    model.rtree_feature_space3D(ax, fontsize, ticks_fontsize, fontname,
                      azim, elev, dist, show, colors, markersize, n_colors_in_map)

def ctreeviz_univar(tree_model,
                    X_train: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                    y_train: (pd.Series, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                    tree_index: int = None,  # required in case of tree ensemble
                    fontsize=10, fontname="Arial", nbins=25, gtype='strip',
                    show={'title', 'legend', 'splits'},
                    colors=None,
                    ax=None):
    warnings.warn("ctreeviz_univar() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.ctree_feature_space(...)",
                  DeprecationWarning, stacklevel=2)

    if isinstance(feature_names, str):
        feature_names = [feature_names]
    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.ctree_feature_space(fontsize, fontname, nbins, gtype, show, colors, ax)


def ctreeviz_bivar(tree_model,
                   X_train: (pd.DataFrame, np.ndarray) = None,  # dataframe with only one column
                   y_train: (pd.Series, np.ndarray) = None,
                   feature_names: List[str] = None,
                   target_name: str = None,
                   class_names: (Mapping[Number, str], List[str]) = None,  # required if classifier,
                   tree_index: int = None,  # required in case of tree ensemble
                   fontsize=10,
                   fontname="Arial",
                   show={'title', 'legend', 'splits'},
                   colors=None,
                   ax=None):
    """
    Show tesselated 2D feature space for bivariate classification tree. X_train can
    have lots of features but features lists indexes of 2 features to train tree with.
    """
    warnings.warn("ctreeviz_bivar() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.ctree_feature_space(...)",
                  DeprecationWarning, stacklevel=2)

    if isinstance(feature_names, str):
        feature_names = [feature_names]
    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.ctree_feature_space(fontsize=fontsize,
                              fontname=fontname,
                              show=show,
                              colors=colors,
                              ax=ax)

def dtreeviz(tree_model,
             X_train: (pd.DataFrame, np.ndarray) = None,
             y_train: (pd.DataFrame, np.ndarray) = None,
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
             title_fontsize: int = 10,
             colors: dict = None,
             scale=1.0
             ) \
        -> DTreeVizRender:
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
                       fit to X_train, y_train.
    :param X_train: A data frame or 2-D matrix of feature vectors used to train the model.
    :param y_train: A pandas Series or 1-D vector with target or classes values. These values should be numeric types.
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
    warnings.warn("dtreeviz() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.view()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    return model.view(precision, orientation,
                      instance_orientation,
                      show_root_edge_labels, show_node_labels, show_just_path, fancy, histtype, highlight_path, X,
                      max_X_features_LR, max_X_features_TD, depth_range_to_display, label_fontsize, ticks_fontsize,
                      fontname, title, title_fontsize, colors=colors, scale=scale)


def viz_leaf_samples(tree_model,
                     X_train: (pd.DataFrame, np.ndarray) = None,
                     feature_names: List[str] = None,
                     tree_index: int = None,  # required in case of tree ensemble
                     display_type: str = "plot",
                     colors: dict = None,
                     fontsize: int = 10,
                     fontname: str = "Arial",
                     grid: bool = False,
                     bins: int = 10,
                     min_samples: int = 0,
                     max_samples: int = None,
                     figsize: tuple = None,
                     ax=None):
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

    Note : If the X_train and y_train are the datasets used to trained the model, then we will investigate the tree model
    as it was trained. We can give other X_train and y_train datasets, ex. validation dataset, to see how the new data is
    spread over the tree.

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param X_train: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
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
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: optional matplotlib "axes" to draw into
    """
    warnings.warn("viz_leaf_samples() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.leaf_sizes()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, None, feature_names, None, None,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.leaf_sizes(display_type, colors, fontsize,
                     fontname, grid, bins, min_samples, max_samples, figsize, ax)


def viz_leaf_criterion(tree_model,
                       tree_index: int = None,  # required in case of tree ensemble,
                       display_type: str = "plot",
                       colors: dict = None,
                       fontsize: int = 10,
                       fontname: str = "Arial",
                       grid: bool = False,
                       bins: int = 10,
                       figsize: tuple = None,
                       ax=None):
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
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: optional matplotlib "axes" to draw into
    :return:
    """
    warnings.warn("viz_leaf_criterion() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.leaf_purity()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, None, None, None, None, None,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.leaf_purity(display_type, colors, fontsize, fontname, grid, bins, figsize, ax)


def ctreeviz_leaf_samples(tree_model,
                          X_train: (pd.DataFrame, np.ndarray) = None,
                          y_train: (pd.DataFrame, np.ndarray) = None,
                          feature_names: List[str] = None,
                          tree_index: int = None,  # required in case of tree ensemble,
                          display_type: str = "plot",
                          plot_ylim: int = None,
                          colors: dict = None,
                          fontsize: int = 10,
                          fontname: str = "Arial",
                          grid: bool = False,
                          figsize: tuple = None,
                          ax=None):
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
    :param X_train: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param y_train: pd.Series, np.ndarray
        Target variable
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
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
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: optional matplotlib "axes" to draw into
    """
    warnings.warn("ctreeviz_leaf_samples() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.ctree_leaf_distributions()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, None, None,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.ctree_leaf_distributions(display_type, plot_ylim, colors, fontsize, fontname, grid, figsize, ax)


def viz_leaf_target(tree_model,
                    X_train: (pd.DataFrame, np.ndarray) = None,
                    y_train: (pd.DataFrame, np.ndarray) = None,
                    feature_names: List[str] = None,
                    target_name: str = None,
                    tree_index: int = None,  # required in case of tree ensemble,
                    show_leaf_labels: bool = True,
                    colors: dict = None,
                    markersize: int = 50,
                    label_fontsize: int = 10,
                    fontname: str = "Arial",
                    precision: int = 1,
                    grid: bool = False,
                    prediction_line_width: int = 2,
                    figsize: tuple = None,
                    ax=None):
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
    :param X_train: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param y_train: pd.Series, np.ndarray
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
    :param grid: bool
        True if we want to display the grid lines on the visualization
    :param prediction_line_width: int
        The width of prediction line.
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: optional matplotlib "axes" to draw into
    """
    warnings.warn("viz_leaf_target() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.rtree_leaf_distributions()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, None,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    model.rtree_leaf_distributions(show_leaf_labels,
                                   colors, markersize, label_fontsize, fontname, precision, grid,
                                   prediction_line_width, figsize, ax)


def describe_node_sample(tree_model,
                         node_id: int,
                         X_train: (pd.DataFrame, np.ndarray) = None,
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
        ex. describe_node_sample(tree_classifier, node_id=1, X_train=dataset[features], feature_names=features)
        - maintain backward compatibility

    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param node_id: int
        Node id to interpret
    :param X_train: pd.DataFrame, np.ndarray
        The dataset based on which we want to make this visualisation.
    :param feature_names: List[str], optional
        The list of feature variable's name
    :param tree_index: int, optional
        Required in case of tree ensemble. Specify the tree index to interpret.
    :return: pd.DataFrame
        Node training samples' stats
    """
    warnings.warn("describe_node_sample() function is deprecated starting from version 2.0. \n "
                  "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.node_stats()",
                  DeprecationWarning, stacklevel=2)

    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, None, feature_names, None, None,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)
    return model.node_stats(node_id)

def explain_prediction_path(tree_model,
                            x: np.ndarray,
                            X_train=None,
                            y_train=None,  # required for XGBoost
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


    :param tree_model: tree.DecisionTreeRegressor, tree.DecisionTreeClassifier, xgboost.core.Booster,
                dtreeviz.models.sklearn_decision_trees.ShadowSKDTree,
                dtreeviz.models.xgb_decision_trees.ShadowXGBDTree
        The tree model or dtreeviz shadow tree model to interpret
    :param x: np.ndarray
        The data instance for which we want to investigate prediction path
    :param y_train: pd.Series, np.ndarray
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
    shadow_tree = ShadowDecTree.get_shadow_tree(tree_model, X_train, y_train, feature_names, target_name, class_names,
                                                tree_index)
    model = DTreeVizAPI(shadow_tree)

    if explanation_type == "sklearn_default":
        warnings.warn(
            "explain_prediction_path(explanation_type='sklearn_default') function is deprecated starting from version 2.0. \n "
            "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.instance_feature_importance()",
            DeprecationWarning, stacklevel=2)
        model.instance_feature_importance(x)
    else:
        warnings.warn("explain_prediction_path() function is deprecated starting from version 2.0. \n "
                      "For the same functionality, please use this code instead: \n m = dtreeviz.model(...) \n m.explain_prediction_path()",
                      DeprecationWarning, stacklevel=2)
        return model.explain_prediction_path(x)
