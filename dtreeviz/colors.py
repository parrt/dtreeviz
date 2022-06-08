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
CATEGORICAL_SPLIT_LEFT= '#FFC300'
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

def get_hex_colors(n_classes,cmap_name = "RdYlBu"):
    """will generate a list of lists that contain n discrete hex colors
     from a given matplotlib colormap based on the number of classes in 
     a given classifier model as determined in trees.py. Defaults to the 
     "RdYlBu" colormap. 

    Args:
        n_classes (int): the number of classes in a classifier model as determined 
        by trees.py
        cmap_name (str, optional): any valid matplotlib colormap. Defaults to "RdYlBu".

    Returns:
        list: a list of lists where each inner list item contains n discrete hex colors.
        Ex:
        get_hex_colors(10)

        [
            None, # 0 classes
            None, # 1 class
            ['#a50026ff', '#313695ff'], # 2 classes
            ['#a50026ff', '#ffffbfff', '#313695ff'], # 3 classes
            ['#a50026ff', '#fdbf71ff', '#bde2eeff', '#313695ff'], # 4 classes
            ['#a50026ff', '#f88d52ff', '#ffffbfff', '#90c3ddff', '#313695ff'], # 5 classes
            ['#a50026ff', '#f46d43ff', '#fee090ff', '#e0f3f8ff', '#74add1ff', '#313695ff'], # 6 classes
            ['#a50026ff', '#ea593aff', '#fdbf71ff', '#ffffbfff', '#bde2eeff', '#649ac7ff', '#313695ff'], # 7 classes
            ['#a50026ff', '#e34a33ff', '#fca55dff', '#fee99dff', '#e9f6e8ff', '#a3d3e6ff', '#598dc0ff', '#313695ff'], # 8 classes
            ['#a50026ff', '#de3f2eff', '#f88d52ff', '#fed384ff', '#ffffbfff', '#d3ecf4ff', '#90c3ddff', '#5183bbff', '#313695ff'], # 9 classes
            ['#a50026ff', '#da372aff', '#f67b4aff', '#fdbf71ff', '#feeea5ff', '#eef8dfff', '#bde2eeff', '#80b7d6ff', '#4a7bb7ff', '#313695ff'], # 10 classes

        ]
    """
 
    
    hex_colors = []
    for i in range(2, n_classes + 1):
        cmap = matplotlib.cm.get_cmap(cmap_name, i)
        hex_colors.append(
            [
                matplotlib.colors.to_hex(rgb, keep_alpha=True)
                for rgb in cmap(np.arange(0, cmap.N))
            ]
        )

    hex_colors.insert(0, None)
    hex_colors.insert(0, None)
    
    return hex_colors


COLORS = {'scatter_edge': GREY,
          'scatter_marker': BLUE,
          'scatter_marker_alpha': 0.7,
          'class_boundary' : GREY,
          'warning' : '#E9130D',
          'tile_alpha':0.8,            # square tiling in clfviz to show probabilities
          'tesselation_alpha': 0.3,    # rectangular regions for decision tree feature space partitioning
          'tesselation_alpha_3D': 0.5,
          'split_line': GREY,
          'mean_line': '#f46d43',
          'axis_label': GREY,
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
          'node_label': GREY,
          'tick_label': GREY,
          'leaf_label': GREY,
          'pie': GREY,
          'hist_bar': LIGHTBLUE,
          'categorical_split_left': CATEGORICAL_SPLIT_LEFT,
          'categorical_split_right': CATEGORICAL_SPLIT_RIGHT
          }


def adjust_colors(colors):
    if colors is None:
        return COLORS
    return dict(COLORS, **colors)
