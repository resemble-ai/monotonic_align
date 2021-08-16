import numpy
from setuptools import Extension, find_packages
from distutils.core import setup
from Cython.Build import cythonize


_VERSION = '1.0'


ext_modules = cythonize(
  "monotonic_align/core.pyx",
  compiler_directives={'language_level' : "3"},
)

setup(
  name='monotonic_align',
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include(), "monotonic_align"],
  packages=find_packages(),
  version=_VERSION,
)
