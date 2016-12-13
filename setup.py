'''
Compiles the fastUpdates module
 to run, enter the code:
python setup.py build_ext --inplace
'''

from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    packages=["cython_library"],
    include_dirs=[np.get_include()],
    ext_modules=cythonize("**/*.pyx"),
)