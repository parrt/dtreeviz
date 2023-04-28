from setuptools import setup, find_packages

# To RELEASE:
#
# $ python3 -m build
# $ twine upload dist/dtreeviz-1.4.0.tar.gz dist/dtreeviz-1.4.0-py3-none-any.whl


extra_xgboost = ['xgboost']
extra_pyspark = ['pyspark']
extra_lightgbm = ['lightgbm']
extra_tensorflow = ['tensorflow_decision_forests']


setup(
    name='dtreeviz',
    version='2.2.2',
    url='https://github.com/parrt/dtreeviz',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'graphviz>=0.9',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'colour',
        'pytest'
    ],
    extras_require={
        'xgboost': extra_xgboost,
        'pyspark': extra_pyspark,
        'lightgbm': extra_lightgbm,
        'tensorflow_decision_forests': extra_tensorflow,
        'all': extra_xgboost + extra_pyspark + extra_lightgbm + extra_tensorflow,
    },
    python_requires='>=3.6',
    author='Terence Parr, Tudor Lapusan, and Prince Grover',
    author_email='parrt@antlr.org',
    description='''A Python 3 library for sci-kit learn, XGBoost, LightGBM, Spark, and TensorFlow decision tree visualization''',
    long_description='''A python library for decision tree visualization and model interpretation.  Decision trees are the fundamental building block of [gradient boosting machines](http://explained.ai/gradient-boosting/index.html) and [Random Forests](https://en.wikipedia.org/wiki/Random_forest)(tm), probably the two most popular machine learning models for structured data.  Visualizing decision trees is a tremendous aid when learning how these models work and when interpreting models. The visualizations are inspired by an educational animation by [R2D3](http://www.r2d3.us/); [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). Please see [How to visualize decision trees](http://explained.ai/decision-tree-viz/index.html) for deeper discussion of our decision tree visualization library and the visual design decisions we made.

Currently dtreeviz supports: [scikit-learn](https://scikit-learn.org/stable), [XGBoost](https://xgboost.readthedocs.io/en/latest), [Spark MLlib](https://spark.apache.org/mllib/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), and [Tensorflow](https://www.tensorflow.org/decision_forests).  See [Installation instructions](README.md#Installation).''',
    keywords='machine-learning data structures trees visualization',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
