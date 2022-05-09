import numpy
from setuptools import Extension, find_packages
from distutils.core import setup
from Cython.Build import cythonize


_VERSION = "1.2"


ext_modules = cythonize(
  ["monotonic_align/core.pyx",
   "monotonic_align/core1alt.pyx",
   "monotonic_align/core2.pyx",
   "monotonic_align/core2eps.pyx",
  ],
  compiler_directives={"language_level": "3"},
)

setup(
  name="monotonic_align",
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include(), "monotonic_align"],
  packages=find_packages(),
  setup_requires=["numpy", "cython"],
  install_requires=["numpy"],
  version=_VERSION,
)
