"""
Prediction path interpretation for decision tree models.
In this moment, it contains "plain english" implementation, but others can be added in the future.
"""
import numpy
import pandas
from sklearn import tree


def explain_prediction_plain_english(tree_model: (tree.DecisionTreeClassifier, tree.DecisionTreeRegressor),
                                     X: (pandas.core.series.Series, numpy.ndarray),
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

    # TODO
    # X is based on dataframe, we need to search into it using feature_name. Find a more general approach.
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


def explain_prediction_weights(tree_model, X):
    return "to be implemented"


def get_prediction_explainer(explanation_type: str):
    """Factory method responsible to return a prediction path implementation based on argument 'explanation_type'

    :param explanation_type: specify the type of path explanation to be returned
    :return: method implementation for specified path explanation.
    """

    if explanation_type == "plain_english":
        return explain_prediction_plain_english
    elif explanation_type == "weights":
        return explain_prediction_weights
    else:
        raise ValueError(f"Explanation type {explanation_type} is not supported yet!")
