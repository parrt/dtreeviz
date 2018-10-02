# dtreeviz -- Decision tree visualization

A python library for decision tree visualization and model interpretation.

By [Terence Parr](http://parrt.cs.usfca.edu) and [Prince Grover](https://www.linkedin.com/in/groverpr)

See [How to visualize decision trees](http://explained.ai/decision-tree-viz/index.html) for deeper discussion of our decision tree visualization library and the visual design decisions we made. 

## Install

(*So far, we've only tested this on OS X*.)  To install (Python >=3.6 only), do this:

```bash
pip install dtreeviz
```

and you need the following tools for the decision tree visualizations to work:

```bash
brew install graphviz --with-librsvg --with-app --with-pango
```

(The `--with-librsvg` is absolutely required.)

Please email us with notes on making it work on other platforms. thanks!

## Discussion

Decision trees are the fundamental building block of [gradient boosting machines](http://explained.ai/gradient-boosting/index.html) and [Random Forests](https://en.wikipedia.org/wiki/Random_forest)(tm), probably the two most popular machine learning models for structured data.  Visualizing decision trees is a tremendous aid when learning how these models work and when interpreting models.  Unfortunately, current visualization packages are rudimentary and not immediately helpful to the novice. For example, we couldn't find a library that visualizes how decision nodes split up the feature space. It is also uncommon for libraries to support visualizing a specific feature vector as it weaves down through a tree's decision nodes; we could only find one image showing this.

So, we've created a general package for [scikit-learn](https://github.com/scikit-learn/scikit-learn) decision tree visualization and model interpretation, which we'll be using heavily in an upcoming [machine learning book](https://mlbook.explained.ai/) (written with [Jeremy Howard](http://www.fast.ai/about/#jeremy)).

The visualizations are inspired by an educational animiation by [R2D3](http://www.r2d3.us/); [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). With `dtreeviz`, you can visualize how the feature space is split up at decision nodes, how the training samples get ditributed in leaf nodes and how the tree makes predictions for a specific observation. These operations are critical to for  understanding how classfication or regression decision trees work. If you're not familiar with decision trees, check out [fast.ai's Introduction to Machine Learning for Coders MOOC](http://course.fast.ai/ml).

## Usage

`dtree`: Main function to create decision tree visualization. Given a decision tree regressor or classifier, creates and returns a tree visualization using the graphviz (DOT) language.

* **Required libraries**:  
Basic libraries and imports that will (might) be needed to generate the sample visualizations shown in examples below. 
 
```bash
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
```

* **Regression decision tree**:   
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
  
  
* **Classification decision tree**:  
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

* **Prediction path**:  
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
  
* **Decision tree without scatterplot or histograms for decision nodes**:  
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
              fancy=False )  # fance=False to remove histograms/scatterpots from decision nodes
              
viz.view() 
```

<img src=testing/samples/breast_cancer-TD-4-simple.svg width=80% height=60%>


For more examples and different implementations, please see the jupyter [notebook](notebooks/examples.ipynb) full of examples.

## Build guidelines

Install anaconda3 on your system.

### Mac prerequisites

We need `dot` executable to convert graphviz files to images.  Make sure to install using:

```bash
brew install graphviz --with-librsvg --with-app --with-pango
```

### Windows 10 prerequisites

We need `dot` executable to convert graphviz files to images.  From "anaconda prompt":

we need .9 graphviz lib and we get 0.8.4 with this:

```bash
conda install -c conda-forge python-graphviz 
```

so let's do `dot` executable first:

```bash
conda install -c anaconda graphviz
```

which updates:

```bash
    ca-certificates: 2018.03.07-0                     --> 2018.03.07-0         anaconda
    certifi:         2018.4.16-py36_0     conda-forge --> 2018.4.16-py36_0     anaconda
    conda:           4.5.11-py36_0        conda-forge --> 4.5.11-py36_0        anaconda
    graphviz:        2.38-hfd603c8_2                  --> 2.38.0-4             anaconda
    openssl:         1.0.2o-h8ea7d77_0                --> 1.0.2o-h8ea7d77_0    anaconda
    qt:              5.9.5-vc14he4a7d60_0             --> 5.9.5-vc14he4a7d60_0 anaconda [vc14]
```

then let's try:

```bash
pip install -U graphviz
```

Yep, it got 0.9 graphviz lib.

Must still add to path even though anaconda prompt sees it. graphviz does not: `C:\Users\parrt\Anaconda3\Library\bin` on my box.  Do we need a reboot after setting env variable `Path`?  Nope. ok, that one is dot.bat. I [downloaded](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) and updated my Path env var but no luck. Another reboot? Actually try this:

```python
>>> import os
>>> os.getenv('Path')
'C:\\Users\\parrt\\Anaconda3;C:\\Users\\parrt\\Anaconda3\\Library\\mingw-w64\\bin;C:\\Users\\parrt\\Anaconda3\\Library\\usr\\bin;C:\\Users\\parrt\\Anaconda3\\Library\\bin;C:\\Users\\parrt\\Anaconda3\\Scripts;C:\\Users\\parrt\\Anaconda3\\bin;C:\\WINDOWS\\system32;C:\\WINDOWS;C:\\WINDOWS\\System32\\Wbem;C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\;C:\\Program Files (x86)\\Windows Kits\\8.1\\Windows Performance Toolkit\\;C:\\Program Files\\Git\\cmd;C:\\Program Files\\dotnet\\;C:\\WINDOWS\\System32\\OpenSSH\\;C:\\Users\\parrt\\AppData\\Local\\Microsoft\\WindowsApps;\\Microsoft.NET\\Framework64\\v3.5;C:\\Program Files\\dotnet\\sdk\\2.0.0\\Sdks;C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319;C:\\Users\\parrt\\Anaconda3\\Library\\bin;C:\\Program Files (x86)\\Graphviz2.38\\bin;'
>>> import subprocess
>>> proc = subprocess.Popen(['dot','-V'])
dot - graphviz version 2.38.0 (20140413.2041)
```

So why doesn't graphviz see this?

```python
import os
import subprocess
proc = subprocess.Popen(['dot','-V'])
print( os.getenv('Path') )
```

or

```python
import graphviz.backend as be
cmd = ["dot", "-V"]
stdout, stderr = be.run(cmd, capture_output=True, check=True, quiet=False)
print( stderr )
```

Wow. that worked. ok. try outside jupyter notebook.  This works!!!

```
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *

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

Except for it looks like .png and the >= is wrong chars.


### Linux (Ubuntu) prerequisites

```bash
sudo apt install graphviz
```

The `view()` method works but inline jupyter doesn't yet.

### Install dtreeviz locally

To push the `dtreeviz` library to your local egg cache (force updates).
 
```bash 
python setup.py install -f
```

E.g., on Terence's box, it add `/Users/parrt/anaconda3/lib/python3.6/site-packages/dtreeviz-0.2-py3.6.egg`.


