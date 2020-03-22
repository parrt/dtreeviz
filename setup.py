from setuptools import setup, find_packages

# python setup.py sdist upload

setup(
    name='dtreeviz',
    version='0.8.2',
    url='https://github.com/parrt/dtreeviz',
    license='MIT',
    packages=find_packages(),
    install_requires=['graphviz>=0.9','pandas','numpy','scikit-learn','matplotlib','colour'],
    python_requires='>=3.6',
    author='Terence Parr and Prince Grover',
    author_email='parrt@cs.usfca.edu',
    description='A Python 3 library for sci-kit learn decision tree visualization',
    keywords='machine-learning data structures trees visualization',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
