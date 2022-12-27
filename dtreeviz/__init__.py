from .version import __version__

# NEW API
# import dtreeviz
# call m = dtreeviz.model(...) then m.view() etc...
from dtreeviz.utils import DTreeVizRender
from dtreeviz.trees import DTreeVizAPI, model

# OLD API
from dtreeviz.compatibility import rtreeviz_univar, \
    rtreeviz_bivar_heatmap, \
    rtreeviz_bivar_3D, \
    ctreeviz_univar, \
    ctreeviz_bivar, \
    dtreeviz, \
    viz_leaf_samples, \
    viz_leaf_criterion, \
    ctreeviz_leaf_samples , \
    viz_leaf_target, \
    describe_node_sample, \
    explain_prediction_path

from dtreeviz.classifiers import decision_boundaries
