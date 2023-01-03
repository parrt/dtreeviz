import os
import re
import tempfile
import warnings
import xml.etree.cElementTree as ET
from pathlib import Path
from sys import platform as PLATFORM

import graphviz
import pandas as pd
from matplotlib import patches as patches
from numpy import ndarray
from numbers import Number
from typing import Tuple, Sequence


def inline_svg_images(svg) -> str:
    """
    Inline IMAGE tag refs in graphviz/dot -> SVG generated files.

    Convert all .svg image tag refs directly under g tags like:

    <g id="node1" class="node">
        <image xlink:href="/tmp/node4.svg" width="45px" height="76px" preserveAspectRatio="xMinYMin meet" x="76" y="-80"/>
    </g>

    to

    <g id="node1" class="node">
        <svg width="45px" height="76px" viewBox="0 0 49.008672 80.826687" preserveAspectRatio="xMinYMin meet" x="76" y="-80">
            XYZ
        </svg>
    </g>


    where XYZ is taken from ref'd svg image file:

    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
      "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <!-- Created with matplotlib (http://matplotlib.org/) -->
    <svg height="80.826687pt" version="1.1" viewBox="0 0 49.008672 80.826687" width="49.008672pt"
         xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        XYZ
    </svg>

    Note that width/height must be taken image ref tag and put onto svg tag. We
    also need the viewBox or it gets clipped a bit.

    :param svg: SVG string with <image/> tags.
    :return: svg with <image/> tags replaced with content of referenced svg image files.
    """
    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.fromstring(svg)
    tree = ET.ElementTree(root)
    parent_map = {c: p for p in tree.iter() for c in p}

    # Find all image tags in document (must use svg namespace)
    image_tags = tree.findall(".//svg:g/svg:image", ns)
    for img in image_tags:
        # load ref'd image and get svg root
        svgfilename = img.attrib["{http://www.w3.org/1999/xlink}href"]
        with open(svgfilename, encoding='UTF-8') as f:
            imgsvg = f.read()
        imgroot = ET.fromstring(imgsvg)
        for k,v in img.attrib.items(): # copy IMAGE tag attributes to svg from image file
            if k not in {"{http://www.w3.org/1999/xlink}href"}:
                imgroot.attrib[k] = v
        # replace IMAGE with SVG tag
        p = parent_map[img]
        # print("BEFORE " + ', '.join([str(c) for c in p]))
        p.append(imgroot)
        p.remove(img)
        # print("AFTER " + ', '.join([str(c) for c in p]))

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    xml_str = ET.tostring(root).decode()
    return xml_str


def get_SVG_shape(svg) -> Tuple[Number,Number,Sequence[Number]]:
    """
    Sample line from SVG file from which we can get w,h,viewBox:
    <svg ... height="382pt" viewBox="0.00 0.00 344.00 382.00" width="344pt">
    Return:
    (344.0, 382.0, [0.0, 0.0, 344.0, 382.0])
    """
    root = ET.fromstring(svg)
    attrs = root.attrib
    viewBox = [float(v) for v in attrs['viewBox'].split(' ')]
    return (float(attrs['width'].strip('pt')),
            float(attrs['height'].strip('pt')),
            viewBox)


def scale_SVG(svg:str, scale:float) -> str:
    """
    Convert:

    <svg ... height="382pt" viewBox="0.00 0.00 344.00 382.00" width="344pt">
    <g class="graph" id="graph0" transform="scale(1 1) rotate(0) translate(4 378)">

    To:

    <svg ... height="191.0" viewBox="0.0 0.0 172.0 191.0" width="172.0">
    <g class="graph" id="graph0" transform="scale(.5 .5) rotate(0) translate(4 378)">
    """
    # Scale bounding box etc...
    w, h, viewBox = get_SVG_shape(svg)
    root = ET.fromstring(svg)
    root.set("width", str(w*scale))
    root.set("height", str(h*scale))
    viewBox[2] *= scale
    viewBox[3] *= scale
    root.set("viewBox", ' '.join([str(v) for v in viewBox]))

    # Deal with graph scale
    ns = {"svg": "http://www.w3.org/2000/svg"}
    graph = root.find(".//svg:g", ns) # get first node, which is graph
    transform = graph.attrib['transform']
    pattern = re.compile(r"scale\([0-9.]+\ [0-9.]+\)")
    scale_str = pattern.search(transform).group()
    transform = transform.replace(scale_str, f'scale({scale} {scale})')
    graph.set("transform", transform)

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    xml_str = ET.tostring(root).decode()
    return xml_str


def myround(v,ndigits=2):
    return format(v, '.' + str(ndigits) + 'f')


def _extract_final_feature_names(pipeline, features):
    """
    Computes the final features names of a :py:mod:`~sklearn.pipeline.Pipeline` used in its last
    component.

    Args:
        pipeline (sklearn.pipeline.Pipeline): A pipeline
        features (list): List of input features to the pipeline

    Returns:
        list: Features names used by the last component
    """
    for component in pipeline[:-1]:
        if hasattr(component, 'get_support'):
            features = [f for f, s in zip(features, component.get_support()) if s]
        if hasattr(component, 'get_feature_names'):
            features = component.get_feature_names(features)

    return features


def _normalize_class_names(class_names, nclasses):
    if class_names is None:
        return {i: f"class {i}" for i in range(nclasses)}
    if isinstance(class_names, dict):
        return class_names
    elif isinstance(class_names, Sequence):
        return {i: n for i, n in enumerate(class_names)}
    elif isinstance(class_names, ndarray):
        return list(class_names)
    else:
        raise Exception(f"class_names must be dict or sequence, not {class_names.__class__.__name__}")


def extract_params_from_pipeline(pipeline, X_train, feature_names):
    """
    Extracts necessary parameters from an :py:class:`sklearn.pipeline.Pipeline` to pass into
    :py:class:`dtreeviz.models.sklearn_decision_trees.ShadowSKDTree`.

    Args:
        pipeline (sklearn.pipeline.Pipeline): An SKlearn pipeline whose last component is a decision tree model.
        X_train (numpy.ndarray): The (X)-input data on which the pipeline was fitted on.
        feature_names (list): List of names of the features in `X_train`.

    Returns:
        tuple: Tuple consisting of the tree model, the transformed input data, and a list of feature
        names used by the model.
    """
    # Pick last element of pipeline
    tree_model = pipeline.steps[-1][1]

    feature_names = _extract_final_feature_names(
        pipeline=pipeline,
        features=feature_names
    )
    X_train = pd.DataFrame(
        data=pipeline[:-1].transform(X_train),
        columns=feature_names
    )
    return tree_model, X_train, feature_names


def check_tree_index(tree_index, nr_of_trees):
    if tree_index is None:
        raise ValueError("You need to pass in a tree_index parameter.")
    if tree_index >= nr_of_trees:
        raise ValueError(f"tree_index parameter should have values between [{0}, {nr_of_trees - 1}].")


class DTreeVizRender:
    """
    This object is constructed from graphviz DOT content and knows how to render and save as SVG.
    """
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
        warnings.warn("DTreeVizRender.view() function is deprecated starting from version 2.0. \n "
                      "Please use display() instead",
                      DeprecationWarning, stacklevel=2)
        self.show()

    def show(self):
        """Pop up a new window to display the (SVG) dtreeview view."""
        svgfilename = self.save_svg()
        graphviz.backend.view(svgfilename)

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
        if graphviz.__version__ <= '0.17':
            graphviz.backend.run(cmd, capture_output=True, check=True, quiet=False)
        else:
            graphviz.backend.execute.run_check(cmd, capture_output=True, check=True, quiet=False)

        if filename.endswith(".svg"):
            # now merge in referenced SVG images to make all-in-one file
            with open(filename, encoding='UTF-8') as f:
                svg = f.read()
            svg = inline_svg_images(svg)
            svg = scale_SVG(svg, self.scale)
            with open(filename, "w", encoding='UTF-8') as f:
                f.write(svg)


if __name__ == '__main__':
    # test rig
    with open("/tmp/t.svg") as f:
        svg = f.read()
        svg2 = scale_SVG(svg, scale=(.8))

    with open("/tmp/u.svg", "w") as f:
        f.write(svg2)


def add_classifier_legend(ax, class_names, class_values, facecolors, target_name,
                          colors, fontsize=10, fontname='Arial'):
    # add boxes for legend
    boxes = []
    for c in class_values:
        box = patches.Rectangle((0, 0), 20, 10, linewidth=.4, edgecolor=colors['rect_edge'],
                                facecolor=facecolors[c], label=class_names[c])
        boxes.append(box)
    leg = ax.legend(handles=boxes,
                    frameon=colors['legend_edge'] is not None,
                    shadow=False,
                    fancybox=colors['legend_edge'] is not None,
                    handletextpad=.35,
                    borderpad=.8,
                    bbox_to_anchor=(1.0, 1.0),
                    edgecolor=colors['legend_edge'])

    leg.set_title(target_name, prop={'size': fontsize,
                                     'weight': 'bold',
                                     'family': fontname})

    leg.get_frame().set_linewidth(.5)
    leg.get_title().set_color(colors['legend_title'])
    leg.get_title().set_fontsize(fontsize)
    leg.get_title().set_fontname(fontname)
    for text in leg.get_texts():
        text.set_color(colors['text'])
        text.set_fontsize(fontsize)


def _format_axes(ax, xlabel, ylabel, colors, fontsize, fontname, ticks_fontsize=None, grid=False):

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontname=fontname, color=colors['axis_label'])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=fontname, color=colors['axis_label'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set(linewidth=.3, color=colors['axis'])

    ax.tick_params(axis='both', which='major', width=.3, labelcolor=colors['tick_label'])
    if ticks_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

    ax.grid(visible=grid)
