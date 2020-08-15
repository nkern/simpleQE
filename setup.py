from setuptools import setup
import os
import ast

# get version from __init__.py
init_file = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 'simpleqe/__init__.py')
with open(init_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        if "__version__" in l:
            version = ast.literal_eval(l.split('=')[1].strip())

setup(
    name            = 'simpleqe',
    version         = version,
    license         = 'MIT',
    description     = 'A simple quadratic estimator',
    author          = 'Nicholas Kern',
    url             = "http://github.com/nkern/simpleQE",
    include_package_data = True,
    packages        = ['simpleqe']
    )