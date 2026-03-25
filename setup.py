from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import sys

install_requires = [
    'numpy',
    'jupyterlab',
    'matplotlib',
    'numba',
    'scipy',
    'typing',
    'pybind11',
]

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp", "/std:c++17"]
    extra_link_args = []
elif sys.platform == "darwin":
    extra_compile_args = [
        "-O3", "-fvisibility=hidden", "-fPIC",
        "-Xpreprocessor", "-fopenmp",
        "-std=c++17",
    ]
    extra_link_args = ["-lomp"]
else:  # linux
    extra_compile_args = [
        "-O3", "-fvisibility=hidden", "-fPIC", "-fopenmp",
        "-march=native",
        "-std=c++17",
    ]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "fastppg",
        sources=[
            "ppg_basis/native/fastppg/py_module.cpp",
        ],
        include_dirs=[
            "ppg_basis/native/fastppg",
            np.get_include(),
        ],
        cxx_std=17,
        define_macros=[
            ("PYBIND11_DETAILED_ERROR_MESSAGES", "1"),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='ppg_basis',
    version='1.0.0',
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    url='',
    license='',
    author='Arjun Putcha',
    author_email='arjunputcha@gmail.com',
    description='Tool for PPG Decomposition using various basis functions'
)