import matplotlib
import numpy as np

YELLOW = '#fefecd'
GREEN = '#cfe2d4'
DARKBLUE = '#313695'
BLUE = '#4575b4'
DARKGREEN = '#006400'
LIGHTORANGE = '#fee090'
LIGHTBLUE = '#a6bddb'
GREY = '#444443'
WEDGE_COLOR = GREY
CATEGORICAL_SPLIT_LEFT = '#FFC300'
CATEGORICAL_SPLIT_RIGHT = BLUE

HIGHLIGHT_COLOR = '#D67C03'

color_blind_friendly_colors = [
    None,  # 0 classes
    None,  # 1 class
    ['#FEFEBB', '#a1dab4'],  # 2 classes
    ['#FEFEBB', '#D9E6F5', '#a1dab4'],  # 3 classes
    ['#FEFEBB', '#D9E6F5', '#a1dab4', LIGHTORANGE],  # 4
    ['#FEFEBB', '#D9E6F5', '#a1dab4', '#41b6c4', LIGHTORANGE],  # 5
    ['#FEFEBB', '#c7e9b4', '#41b6c4', '#2c7fb8', LIGHTORANGE, '#f46d43'],  # 6
    ['#FEFEBB', '#c7e9b4', '#7fcdbb', '#41b6c4', '#225ea8', '#fdae61', '#f46d43'],  # 7
    ['#FEFEBB', '#edf8b1', '#c7e9b4', '#7fcdbb', '#1d91c0', '#225ea8', '#fdae61', '#f46d43'],  # 8
    ['#FEFEBB', '#c7e9b4', '#41b6c4', '#74add1', BLUE, DARKBLUE, LIGHTORANGE, '#fdae61', '#f46d43'],  # 9
    ['#FEFEBB', '#c7e9b4', '#41b6c4', '#74add1', BLUE, DARKBLUE, LIGHTORANGE, '#fdae61', '#f46d43', '#d73027']  # 10

]

mpl_colors = [
        None,  # 0 classes
        None,  # 1 class
    ] + [
        [f'C{i}' for i in range(0, n_classes)] for n_classes in range(2, 11)
    ]


def get_hex_colors(n_classes, cmap_name="RdYlBu"):
    """
    Will generate a list of lists that contain n discrete hex colors
    from a given matplotlib colormap based on the number of classes in
    a given classifier model as determined in trees.py. Defaults to the
    "RdYlBu" colormap.

    For backward compatibility with the color_blind_friendly_colors, the first 10 lists will be populated with values
    from color_blind_friendly_colors list.

    Args:
        n_classes (int): the number of classes in a classifier model as determined 
        by trees.py
        cmap_name (str, optional): any valid matplotlib colormap. Defaults to "RdYlBu".

    Returns:
        list: a list of lists where each inner list item contains n discrete hex colors.
    """

    hex_colors = color_blind_friendly_colors.copy()
    if n_classes:
        for i in range(len(color_blind_friendly_colors), n_classes + 1):
            cmap = matplotlib.cm.get_cmap(cmap_name, i)
            hex_colors.append(
                [
                    matplotlib.colors.to_hex(rgb, keep_alpha=True)
                    for rgb in cmap(np.arange(0, cmap.N))
                ]
            )

    return hex_colors


COLORS = {'scatter_edge': GREY,
          'scatter_marker': BLUE,
          'scatter_marker_alpha': 0.7,
          'class_boundary': GREY,
          'warning': '#E9130D',
          'tile_alpha': 0.8,  # square tiling in decision_boundaries to show probabilities
          'tessellation_alpha': 0.3,  # rectangular regions for decision tree feature space partitioning
          'tessellation_alpha_3D': 0.5,
          'split_line': GREY,
          'mean_line': '#f46d43',
          'axis_label': GREY,
          'axis': GREY,
          'title': GREY,
          'legend_title': GREY,
          'legend_edge': GREY,
          'edge': GREY,
          'color_map_min': '#c7e9b4',
          'color_map_max': '#081d58',
          'classes': color_blind_friendly_colors,
          'rect_edge': GREY,
          'text': GREY,
          'highlight': HIGHLIGHT_COLOR,
          'wedge': WEDGE_COLOR,
          'text_wedge': WEDGE_COLOR,
          'arrow': GREY,
          'larrow': GREY,
          'rarrow': GREY,
          'node_label': GREY,
          'tick_label': GREY,
          'leaf_label': GREY,
          'pie': GREY,
          'hist_bar': LIGHTBLUE,
          'categorical_split_left': CATEGORICAL_SPLIT_LEFT,
          'categorical_split_right': CATEGORICAL_SPLIT_RIGHT
          }


def adjust_colors(colors, n_classes=None, cmp="RdYlBu"):
    if colors is None:
        if n_classes and n_classes > len(color_blind_friendly_colors) - 1:
            # in case the number of classes is bigger than the color_blind_friendly_colors can handle, we will add more
            # color class values
            COLORS["classes"] = get_hex_colors(n_classes, cmp)
        return COLORS

    return dict(COLORS, **colors)
