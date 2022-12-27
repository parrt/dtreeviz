from setuptools import setup, find_packages

# To RELEASE:
#
# $ python3 -m build
# $ twine upload dist/dtreeviz-1.4.0.tar.gz dist/dtreeviz-1.4.0-py3-none-any.whl

setup(
    name='dtreeviz',
    version='2.0.0',
    url='https://github.com/parrt/dtreeviz',
    license='MIT',
    packages=find_packages(),
    install_requires=['graphviz>=0.9','pandas','numpy','scikit-learn',
                        'matplotlib','colour', 'pytest'],
    extras_require={'xgboost': ['xgboost'],
                    'pyspark':['pyspark'],
                    'lightgbm':['lightgbm'],
                    'tensorflow_decision_forests':['tensorflow_decision_forests']},
    python_requires='>=3.6',
    author='Terence Parr, Tudor Lapusan, and Prince Grover',
    author_email='parrt@antlr.org',
    description='A Python 3 library for sci-kit learn, XGBoost, LightGBM, Spark, and TensorFlow decision tree visualization',
    long_description='README.md',
    keywords='machine-learning data structures trees visualization',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
