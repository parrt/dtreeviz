# dtreeviz : Decision Tree Visualization

## Description
A python library for decision tree visualization and model interpretation.

By [Terence Parr](http://parrt.cs.usfca.edu) and [Prince Grover](https://www.linkedin.com/in/groverpr)

See [How to visualize decision trees](http://explained.ai/decision-tree-viz/index.html) for deeper discussion of our decision tree visualization library and the visual design decisions we made. 

## Discussion

Decision trees are the fundamental building block of [gradient boosting machines](http://explained.ai/gradient-boosting/index.html) and [Random Forests](https://en.wikipedia.org/wiki/Random_forest)(tm), probably the two most popular machine learning models for structured data.  Visualizing decision trees is a tremendous aid when learning how these models work and when interpreting models.  Unfortunately, current visualization packages are rudimentary and not immediately helpful to the novice. For example, we couldn't find a library that visualizes how decision nodes split up the feature space. It is also uncommon for libraries to support visualizing a specific feature vector as it weaves down through a tree's decision nodes; we could only find one image showing this.

So, we've created a general package for [scikit-learn](https://github.com/scikit-learn/scikit-learn) decision tree visualization and model interpretation, which we'll be using heavily in an upcoming [machine learning book](https://mlbook.explained.ai/) (written with [Jeremy Howard](http://www.fast.ai/about/#jeremy)).

The visualizations are inspired by an educational animation by [R2D3](http://www.r2d3.us/); [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). With `dtreeviz`, you can visualize how the feature space is split up at decision nodes, how the training samples get distributed in leaf nodes and how the tree makes predictions for a specific observation. These operations are critical to for  understanding how classification or regression decision trees work. If you're not familiar with decision trees, check out [fast.ai's Introduction to Machine Learning for Coders MOOC](http://course.fast.ai/ml).

## Install

Install anaconda3 on your system.

To install (Python >=3.6 only), do this (from Anaconda Prompt on Windows!):

```bash
pip install dtreeviz
```

This should also pull in the `graphviz` Python library (>=0.9), which we are using for platform specific stuff.

Please email [Terence](mailto:parrt@cs.usfca.edu) with any helpful notes on making dtreeviz work (better) on other platforms. Thanks! 

For your specific platform, please see the following subsections.

### Mac

*The brew spec for graphviz recently removed the options we use to use, --with-librsvg --with-pango. Now you have to build graphviz from source to get this working. (Feb 4, 2018)*

Make sure to have the latest XCode installed and command-line tools installed. You can run `xcode-select --install` from the command-line to install those if XCode is already installed. You also have to sign the XCode license agreement, which you can do with `sudo xcodebuild -license` from command-line. The brew install shown next needs to build graphviz, so you need XCode set up properly.

You need the graphviz binary for `dot` installed with librsvg and pango. Make sure you follow this procedure (verified on three recent macs with 10.13) to build graphviz 2.40.1 from source and install:

```bash
brew uninstall graphviz
brew reinstall pango librsvg  # even if already there, please reinstall

cd /tmp
wget https://graphviz.gitlab.io/pub/graphviz/stable/SOURCES/graphviz.tar.gz
tar xvfz graphviz.tar.gz
cd graphviz-2.40.1/

rm -rf /usr/local/lib/graphviz # in case old stuff is there
# make sure to run ./configure after reinstalling pango, librsvg
./configure --includedir=/usr/local/include/graphviz
make -j 8 # 8 threads
make install
```

Just to be sure, remove `dot` from any anaconda installation, for example:

```bash
rm ~/anaconda3/bin/dot
```

From command line, this command

```bash
dot -Tsvg:cairo
```

should work, in the sense that it just stares at you without giving an error. You can hit control-C to escape back to the shell. If you still get an error message that says something about trying `-Tsvg:svg:core` option, then we still have a problem. Make sure that you are using the right `dot`:

```bash
$ which dot
/usr/local/bin/dot
$ ls -l $(which dot)
-rwxr-xr-x  1 parrt  wheel  22920 Feb  5 09:02 /usr/local/bin/dot*
$
```

If instead, it says something like

```
$ ls -l $(which dot)
lrwxr-xr-x  1 parrt  wheel  33 Feb  4 19:54 /usr/local/bin/dot@ -> ../Cellar/graphviz/2.40.1/bin/dot
```

then you're still using the brew version. Do a `brew uninstall graphviz` and then do `make install` again from the `/tmp/graphviz-2.40.1` dir.
 
The OS X version is able to generate/save images in any format `dot` is allowed to use with the `-T{format}:cairo` option. So .svg, .pdf are totally safe bets.

**Limitations.** Jupyter notebook has a bug where they do not show .svg files correctly, but Juypter Lab has no problem.

### Linux (Ubuntu 18.04)

To get the `dot` binary do:
 
```bash
sudo apt install graphviz
```

**Limitations.** The `view()` method works to pop up a new window and images appear inline for jupyter notebook but not jupyter lab (It gets an error parsing the SVG XML.)  The notebook images also have a font substitution from the Arial we use and so some text overlaps. Only .svg files can be generated on this platform.

### Windows 10

[Download graphviz-2.38.msi](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) and update your `Path` environment variable. It's windows so you might need a reboot after updating that environment variable.  You should see this from the Anaconda Prompt:

```
(base) C:\Users\Terence Parr>where dot
C:\Program Files (x86)\Graphviz2.38\bin\dot.exe
```

(Do not use `conda install -c conda-forge python-graphviz` as you get an old version of `graphviz` python library.)

Verify from the Anaconda Prompt that this works:

```
dot -V
```

If it doesn't work, you have a `Path` problem. I found the following test programs useful. The first one sees if Python can find `dot`:

```python
import os
import subprocess
proc = subprocess.Popen(['dot','-V'])
print( os.getenv('Path') )
```

The following version does the same thing except uses `graphviz` Python libraries backend support utilities, which is what we use in dtreeviz:

```python
import graphviz.backend as be
cmd = ["dot", "-V"]
stdout, stderr = be.run(cmd, capture_output=True, check=True, quiet=False)
print( stderr )
```

Jupyter Lab and Jupyter notebook both show the inline .svg images well.

### Verify graphviz installation

Try making file `t.dot` with content `digraph T { A -> B }` and then running this from the command line:

```
dot -Tsvg:cairo -o t.svg t.dot
```

That should give a simple `t.svg` file that opens properly.  If you get errors from `dot`, it will not work from the dtreeviz python code.  If it can't find `dot` then you didn't update your `PATH` environment variable or there is some other install issue with `graphviz`.

### Limitations

Finally, don't use IE to view .svg files. Use Edge as they look much better. I suspect that IE is displaying them as a rasterized not vector images. Only .svg files can be generated on this platform.

## Usage

`dtree`: Main function to create decision tree visualization. Given a decision tree regressor or classifier, creates and returns a tree visualization using the graphviz (DOT) language.

### Required libraries

Basic libraries and imports that will (might) be needed to generate the sample visualizations shown in examples below. 
 
```bash
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
```

### Regression decision tree
The default orientation of tree is top down but you can change it to left to right using `orientation="LR"`. `view()` gives a pop up window with rendered graphviz object. 

```bash
regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr.fit(boston.data, boston.target)

viz = dtreeviz(regr,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names)
              
viz.view()              
```
  
<img src=testing/samples/boston-TD-2.svg width=60% height=40%>
  
  
### Classification decision tree
An additional argument of `class_names` giving a mapping of class value with class name is required for classification trees. 

```bash
classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = load_iris()
classifier.fit(iris.data, iris.target)

viz = dtreeviz(classifier, 
               iris.data, 
               iris.target,
               target_name='variety',
              feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
              
viz.view() 
```

<img src=testing/samples/iris-TD-2.svg width=50% height=30% align="center">

### Prediction path
Highlights the decision nodes in which the feature value of single observation passed in argument `X` falls. Gives feature values of the observation and highlights features which are used by tree to traverse path. 
  
```bash
regr = tree.DecisionTreeRegressor(max_depth=2)  # limit depth of tree
diabetes = load_diabetes()
regr.fit(diabetes.data, diabetes.target)
X = diabetes.data[np.random.randint(0, len(diabetes.data)),:]  # random sample from training

viz = dtreeviz(regr,
               diabetes.data, 
               diabetes.target, 
               target_name='value', 
               orientation ='LR',  # left-right orientation
               feature_names=diabetes.feature_names,
               X=X)  # need to give single observation for prediction
              
viz.view()  
```
<img src=testing/samples/diabetes-LR-2-X.svg width=100% height=50%>
  
### Decision tree without scatterplot or histograms for decision nodes
Simple tree without histograms or scatterplots for decision nodes. 
Use argument `fancy=False`  
  
```bash
classifier = tree.DecisionTreeClassifier(max_depth=4)  # limit depth of tree
cancer = load_breast_cancer()
classifier.fit(cancer.data, cancer.target)

viz = dtreeviz(classifier,
              cancer.data,
              cancer.target,
              target_name='cancer',
              feature_names=cancer.feature_names, 
              class_names=["malignant", "benign"],
              fancy=False )  # fance=False to remove histograms/scatterplots from decision nodes
              
viz.view() 
```

<img src=testing/samples/breast_cancer-TD-4-simple.svg width=80% height=60%>

For more examples and different implementations, please see the jupyter [notebook](notebooks/examples.ipynb) full of examples.

### Regression univariate feature-target space

<img src="https://user-images.githubusercontent.com/178777/49105092-9b264d80-f234-11e8-9d67-cc58c47016ca.png" width="60%">

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X_train, y_train = df_cars.drop('MPG', axis=1), df_cars['MPG']

fig = plt.figure()
ax = fig.gca()
t = rtreeviz_univar(ax,
                    X_train.WGT, y_train,
                    max_depth=2,
                    feature_name='Vehicle Weight',
                    target_name='MPG',
                    fontsize=14)
plt.show()
```

### Regression bivariate feature-target space

<img src="https://user-images.githubusercontent.com/178777/49104999-4edb0d80-f234-11e8-9010-73b7c0ba5fb9.png" width="60%">

```python
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']

features = [2, 1]
X = X.values[:,features]
figsize = (6,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')

t = rtreeviz_bivar_3D(ax,
                      X, y,
                      max_depth=4,
                      feature_names=['Vehicle Weight', 'Horse Power'],
                      target_name='MPG',
                      fontsize=14,
                      elev=20,
                      azim=25,
                      dist=8.2,
                      show={'splits','title'})
plt.show()
```

### Regression bivariate feature-target space heatmap

<img src="https://user-images.githubusercontent.com/178777/49107627-08d57800-f23b-11e8-85a2-ab5894055092.png" width="60%">

```python
from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']

features=[2, 1]
X = X.values[:, features]
figsize = (6, 5)
fig, ax = plt.subplots(1, 1, figsize=figsize)
t = rtreeviz_bivar_heatmap(ax,
                           X, y,
                           max_depth=4,
                           feature_names=['Vehicle Weight', 'Horse Power'],
                           fontsize=14)
plt.show()
```

### Classification univariate feature-target space

<img src="https://user-images.githubusercontent.com/178777/49105084-9497d600-f234-11e8-9097-56835558c1a6.png" width="60%">

```python
from dtreeviz.trees import *

know = pd.read_csv("data/knowledge.csv")
class_names = ['very_low', 'Low', 'Middle', 'High']
know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(class_names)})

x_train = know.PEG
y_train = know['UNS']
figsize = (6,2)
fig, ax = plt.subplots(1, 1, figsize=figsize)
ct = ctreeviz_univar(ax, x_train, y_train, max_depth=3,
                     feature_name = 'PEG', class_names=class_names,
                     target_name='Knowledge',
                     nbins=40, gtype='strip',
                     show={'splits','title'})
plt.tight_layout()
plt.show()
```

### Classification bivariate feature-target space

<img src="https://user-images.githubusercontent.com/178777/49105085-9792c680-f234-11e8-8af5-bc2fde950ab1.png" width="60%">

```python
from dtreeviz.trees import *

know = pd.read_csv("data/knowledge.csv")
class_names = ['very_low', 'Low', 'Middle', 'High']
know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(class_names)})

features=[4,3]
X_train = know.drop('UNS', axis=1)
y_train = know['UNS']
X_train = X_train.values[:, features]
figsize = (6,5)
fig, ax = plt.subplots(1, 1, figsize=figsize)
ctreeviz_bivar(ax, X_train, y_train, max_depth=3,
               feature_names = ['PEG','LPR'],
               class_names=class_names,
               target_name='Knowledge')
plt.tight_layout()
plt.show()
```

## Install dtreeviz locally

Make sure to follow the install guidelines above.

To push the `dtreeviz` library to your local egg cache (force updates) during development, do this (from anaconda prompt on Windows):
 
```bash 
python setup.py install -f
```

E.g., on Terence's box, it add `/Users/parrt/anaconda3/lib/python3.6/site-packages/dtreeviz-0.3-py3.6.egg`.


## Useful Resources

* [How to visualize decision trees](http://explained.ai/decision-tree-viz/index.html)
* [How to explain gradient boosting](http://explained.ai/gradient-boosting/index.html)
* [The Mechanics of Machine Learning](https://mlbook.explained.ai/)
* [Animation by R2D3](http://www.r2d3.us/)
* [A visual introductionn to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [fast.ai's Introduction to Machine Learning for Coders MOOC](http://course.fast.ai/ml)
* Stef van den Elzen's [Interactive Construction, Analysis and
Visualization of Decision Trees](http://alexandria.tue.nl/extra1/afstversl/wsk-i/elzen2011.pdf)
* Some similar feature-space visualizations in [Towards an effective cooperation of the user and the computer for classification, SIGKDD 2000](https://github.com/EE2dev/publications/blob/master/cooperativeClassification.pdf)
* [Beautiful Decisions: Inside BigML’s Decision Trees](https://blog.bigml.com/2012/01/23/beautiful-decisions-inside-bigmls-decision-trees/)
* "SunBurst" approach to tree visualization: [An evaluation of space-filling information visualizations
for depicting hierarchical structures](https://www.cc.gatech.edu/~john.stasko/papers/ijhcs00.pdf)

## Authors

* [**Terence Parr**](http://parrt.cs.usfca.edu/) 
* [**Prince Grover**](https://www.linkedin.com/in/groverpr/)

See also the list of [contributors](https://github.com/parrt/dtreeviz/graphs/contributors) who participated in this project.

## License

This project is licensed under the terms of the MIT license, see [LICENSE](LICENSE).



