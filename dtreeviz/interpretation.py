"""
Prediction path interpretation for decision tree models.
In this moment, it contains "plain english" implementation, but others can be added in the future.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dtreeviz.colors import adjust_colors
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.utils import _format_axes


def explain_prediction_plain_english(shadow_tree: ShadowDecTree,
                                     x: (pd.core.series.Series, np.ndarray)) -> str:
    """
    Explains the prediction path using feature value's range.

    A possible output for this method could be :
        1.5 <= Pclass
        3.5 <= Age < 44.5
        7.91 <= Fare < 54.25
        0.5 <= Sex_label
        Cabin_label < 3.5
        0.5 <= Embarked_label
    Output explanation :
        The model chose to make this prediction because instance's Pclass feature value is bigger or equal to 1.5, Age
        is between 3.5 and 44.5, Fare is between 7.91 and 54.25, and so on.

    :param shadow_tree: tree used to make prediction
    :param x: Instance example to make prediction
    :return: str
        Prediction path explanation in plain english.
    """
    node_feature_index = shadow_tree.get_features()
    feature_names = shadow_tree.feature_names
    node_threshold = shadow_tree.get_thresholds()
    decision_node_path = shadow_tree.predict_path(x)

    # TODO - refactor this logic and find a way to make it simpler
    feature_smaller_values = {}
    feature_bigger_values = {}
    feature_categorical_value = defaultdict(lambda: set())
    feature_categorical_value_not_in = defaultdict(lambda: set())

    for i, node in enumerate(decision_node_path):
        if i == len(decision_node_path) - 1:
            break  # stop at leaf node
        node_id = node.id

        feature_name = feature_names[node_feature_index[node_id]]
        feature_value = x[node_feature_index[node_id]]

        if not shadow_tree.is_categorical_split(node_id):
            feature_split_value = round(node_threshold[node_id], 2)

            if feature_split_value <= feature_value:
                if feature_smaller_values.get(feature_name) is None:
                    feature_smaller_values[feature_name] = []
                feature_smaller_values.get(feature_name).append(feature_split_value)
            elif feature_split_value > feature_value:
                if feature_bigger_values.get(feature_name) is None:
                    feature_bigger_values[feature_name] = []
                feature_bigger_values.get(feature_name).append(feature_split_value)
        else:
            if feature_value in node_threshold[node_id]:
                feature_categorical_value[feature_name].update(node_threshold[node_id])
            else:
                feature_categorical_value_not_in[feature_name].update(node_threshold[node_id])

    prediction_path_output = ""
    for feature_name in feature_names:
        feature_range = ""
        if feature_name in feature_smaller_values:
            feature_range = f"{max(feature_smaller_values[feature_name])} <= {feature_name} "
        if feature_name in feature_bigger_values:
            if feature_range == "":
                feature_range = f"{feature_name} < {min(feature_bigger_values[feature_name])}"
            else:
                feature_range += f" < {min(feature_bigger_values[feature_name])}"

        if feature_range != "":
            prediction_path_output += feature_range + "\n"

    for feature_name in set(list(feature_categorical_value.keys()) + list(feature_categorical_value_not_in.keys())):
        prediction_path_output += f"{feature_name}{' in ' + str(feature_categorical_value[feature_name]) if feature_name in feature_categorical_value else ''}" \
                                  f"{' not in ' + str(feature_categorical_value_not_in[feature_name]) if feature_name in feature_categorical_value_not_in else ''}  \n"

    return prediction_path_output


def explain_prediction_sklearn_default(shadow_tree: ShadowDecTree,
                                       x: (pd.core.series.Series, np.ndarray),
                                       colors: dict = None,
                                       fontsize: int = 10,
                                       fontname: str = "Arial",
                                       grid: bool = False,
                                       figsize: tuple = None,
                                       ax=None):
    """
    Explain prediction calculating features importance using sklearn default algorithm : mean decrease in impurity
    (or gini importance) mechanism.
    This mechanism can be biased, especially for situations where features vary in their scale of measurement or
    their number of categories.
    For more details, you can read this article : https://explained.ai/rf-importance/index.html

    :param shadow_tree: tree used to make prediction
    :param x: Instance example to make prediction
    :param colors: dict, optional
        The set of colors used for plotting
    :param fontsize: int, optional
        Plot labels fontsize
    :param fontname: str, optional
        Plot labels font name
    :param grid: bool
        True if we want to display the grid lines on the visualization
    :param figsize: optional (width, height) in inches for the entire plot
    :param ax: optional matplotlib "axes" to draw into
    """
    decision_node_path = shadow_tree.predict_path(x)
    decision_node_path = [node.id for node in decision_node_path]

    feature_path_importance = shadow_tree.get_feature_path_importance(decision_node_path)

    colors = adjust_colors(colors)
    if ax is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

    df = pd.DataFrame()
    df['features'] = shadow_tree.feature_names
    df['imp'] = feature_path_importance
    df = df.sort_values('imp', ascending=True)

    barcontainers = ax.barh(y=df['features'],
                            width=df['imp'],
                            color=colors["hist_bar"],
                            lw=.3,
                            align='center',
                            height=1)
    for rect in barcontainers.patches:
        rect.set_linewidth(.5)
        rect.set_edgecolor(colors['rect_edge'])

    _format_axes(ax, "Feature Importance", "Features", colors, fontsize, fontname, grid=grid)
