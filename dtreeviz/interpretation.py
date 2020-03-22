"""
Prediction path interpretation for decision tree models.
In this moment, it contains "plain english" implementation, but others can be added in the future.
"""
import numpy as np
import pandas
from sklearn import tree
import matplotlib.pyplot as plt
from dtreeviz.colors import adjust_colors


def explain_prediction_plain_english(tree_model: (tree.DecisionTreeClassifier, tree.DecisionTreeRegressor),
                                     X: (pandas.core.series.Series, np.ndarray),
                                     feature_names):
    """
    Explains the prediction path using feature value's range.

    A possible output for this method could be :
        1.5 <= Pclass(3.0)
        3.5 <= Age(29.7) < 44.5
        7.91 <= Fare(8.05) < 54.25
        0.5 <= Sex_label(1.0)
        Cabin_label(-1.0) < 3.5
        0.5 <= Embarked_label(2.0)
    Output explanation :
        The model chose to make this prediction because instance's Pclass feature value is bigger or equal to 1.5, Age
        is between 3.5 and 44.5, Fare is between 7.91 and 54.25, and so on.

    :param tree_model: tree used to make prediction
    :param X: Instance example to make prediction
    :param feature_names: feature name list
    :return: str
        Prediction path explanation in plain english.
    """

    node_feature_index = tree_model.tree_.feature
    node_threshold = tree_model.tree_.threshold

    node_indicator = tree_model.decision_path([X])
    decision_node_path = node_indicator.indices[node_indicator.indptr[0]:
                                                node_indicator.indptr[1]]
    feature_min_range = {}
    feature_max_range = {}
    for i, node_id in enumerate(decision_node_path):
        if i == len(decision_node_path) - 1:
            break  # stop at leaf node

        feature_name = feature_names[node_feature_index[node_id]]
        feature_value = X[node_feature_index[node_id]]
        feature_split_value = round(node_threshold[node_id], 2)

        if feature_min_range.get(feature_name, feature_value) >= feature_split_value:
            feature_min_range[feature_name] = feature_split_value
        elif feature_max_range.get(feature_name, feature_value) < feature_split_value:
            feature_max_range[feature_name] = feature_split_value

    for feature_name in feature_names:
        feature_range = ""
        if feature_name in feature_min_range:
            feature_range = f"{feature_min_range[feature_name]} <= {feature_name}"
        if feature_name in feature_max_range:
            if feature_range == "":
                feature_range = f"{feature_name} < {feature_max_range[feature_name]}"
            else:
                feature_range += f" < {feature_max_range[feature_name]}"

        if feature_range != "":
            print(feature_range)


def explain_prediction_sklearn_default(tree_model, X, features,
                                       figsize: tuple = (10, 5),
                                       colors: dict = None,
                                       fontsize: int = 14,
                                       fontname: str = "Arial",
                                       grid: bool = False):
    """
    Explain prediction calculating features importance using sklearn default algorithm : mean decrease in impurity
    (or gini importance) mechanism.
    This mechanism can be biased, especially for situations where features vary in their scale of measurement or
    their number of categories.
    For more details, you can read this article : https://explained.ai/rf-importance/index.html

    :param tree_model: tree used to make prediction
    :param X: Instance example to make prediction
    :param features: list
        Feature name list
    :param figsize: tuple of int, optional
        The plot size
    :param colors: dict, optional
        The set of colors used for plotting
    :param fontsize: int, optional
        Plot labels fontsize
    :param fontname: str, optional
        Plot labels font name
    :return:
        Prediction feature's importance plot
    """

    node_indicator = tree_model.decision_path([X])
    decision_node_path = node_indicator.indices[node_indicator.indptr[0]:
                                                node_indicator.indptr[1]]
    feature_path_importance = _get_feature_path_importance_sklearn(tree_model, decision_node_path)
    return _get_feature_path_importance_sklearn_plot(features, feature_path_importance, figsize, colors, fontsize,
                                                     fontname,
                                                     grid)


def _get_feature_path_importance_sklearn_plot(features, feature_path_importance, figsize, colors, fontsize, fontname,
                                              grid):
    colors = adjust_colors(colors)
    fig, ax = plt.subplots(figsize=figsize);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.3)
    ax.spines['bottom'].set_linewidth(.3)
    ax.set_xticks(range(0, len(features)))
    ax.set_xticklabels(features)

    barcontainers = ax.bar(range(0, len(features)), feature_path_importance, color=colors["hist_bar"], lw=.3,
                           align='center',
                           width=1)
    for rect in barcontainers.patches:
        rect.set_linewidth(.5)
        rect.set_edgecolor(colors['rect_edge'])
    ax.set_xlabel("features", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.set_ylabel("feature importance", fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    ax.grid(b=grid)

    return ax


def _get_feature_path_importance_sklearn(tree_model, node_list):
    gini_importance = np.zeros(tree_model.tree_.n_features)
    for node in node_list:
        if tree_model.tree_.children_left[node] != -1:
            node_left = tree_model.tree_.children_left[node]
            node_right = tree_model.tree_.children_right[node]

            gini_importance[tree_model.tree_.feature[node]] += tree_model.tree_.weighted_n_node_samples[node] * \
                                                               tree_model.tree_.impurity[node] \
                                                               - tree_model.tree_.weighted_n_node_samples[node_left] * \
                                                               tree_model.tree_.impurity[node_left] \
                                                               - tree_model.tree_.weighted_n_node_samples[node_right] * \
                                                               tree_model.tree_.impurity[node_right]
    normalizer = np.sum(gini_importance)
    if normalizer > 0.0:
        gini_importance /= normalizer

    return gini_importance


def get_prediction_explainer(explanation_type: str):
    """Factory method responsible to return a prediction path implementation based on argument 'explanation_type'

    :param explanation_type: specify the type of path explanation to be returned
    :return: method implementation for specified path explanation.
    """

    if explanation_type == "plain_english":
        return explain_prediction_plain_english
    elif explanation_type == "sklearn_default":
        return explain_prediction_sklearn_default
    else:
        raise ValueError(f"Explanation type {explanation_type} is not supported yet!")
