"""
Build C binary code using Cython library
"""
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
setup(
    name="Graph utilities",
    ext_modules=cythonize('graph_utils.pyx'),
    include_dirs=[np.get_include()]
)
