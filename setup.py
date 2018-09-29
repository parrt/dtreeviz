from setuptools import setup, find_packages

setup(
    name='animl',
    version='0.1.1',
    url='https://github.com/parrt/animl',
    license='MIT',
    packages=find_packages(),
    install_requires=['graphviz','pandas','numpy','scikit-learn','matplotlib'],
    python_requires='>=3.6',
    author='Terence Parr',
    author_email='parrt@antlr.org',
    description='A python machine learning library for structured data with decision tree visualization',
    keywords='machine-learning data structures',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
