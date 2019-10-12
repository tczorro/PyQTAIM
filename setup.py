#!/usr/bin/env python
from setuptools import find_packages, setup, dist, Extension

dist.Distribution().fetch_build_eggs(["Cython>=0.29", "numpy>=1.16"])

import numpy


try:
    from Cython.Build import cythonize

    extensions = [
        Extension("pyqtaim.pyqtaim", ["src/pyqtaim/pyqtaim.pyx"]),
        Extension("pyqtaim.uniformgrid", ["src/pyqtaim/uniformgrid.pyx"]),
    ]
    ext_modules = cythonize(extensions)
except ValueError:
    ext_modules = [
        Extension("pyqtaim.pyqtaim", ["src/pyqtaim/pyqtaim.c"]),
        Extension("pyqtaim.uniformgrid", ["src/pyqtaim/uniformgrid.c"]),
    ]

setup(
    name="pyqtaim",
    version="0.0.1",
    description="QTAIM.",
    auther="Derrick Yang",
    author_email="yxt1991@gmail.com",
    url="https://github.com/tczorro/pyqtaim.git",
    package_dir={"": "src"},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=1.16", "cython>=0.29"],
)
