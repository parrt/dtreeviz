from setuptools import setup, find_packages

# python setup.py sdist upload

setup(
    name='dtreeviz',
    version='1.3.1',
    url='https://github.com/parrt/dtreeviz',
    license='MIT',
    packages=find_packages(),
    install_requires=['graphviz>=0.9','pandas','numpy','scikit-learn',
                        'matplotlib','colour', 'pytest'],
    extras_require={'xgboost': ['xgboost'], 'pyspark':['pyspark'], 'lightgbm':['lightgbm']},
    python_requires='>=3.6',
    author='Terence Parr, Tudor Lapusan, and Prince Grover',
    author_email='parrt@cs.usfca.edu',
    description='A Python 3 library for sci-kit learn, XGBoost, LightGBM, and Spark decision tree visualization',
    keywords='machine-learning data structures trees visualization',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
