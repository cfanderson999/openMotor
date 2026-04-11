from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import multiprocessing
import platform

try:
    from pyqt_distutils.build_ui import build_ui
    cmdclass = {'build_ui': build_ui}
except ImportError:
    print('pyqt_distutils not found, build_ui command will be unavailable')
    build_ui = None  # user won't have pyqt_distutils when deploying
    cmdclass = {}

try:
    from uilib.fileIO import appVersionStr
except ImportError:
    print('App version not available, defaulting to 0.0.0')
    appVersionStr = '0.0.0'

# OpenMP compiler / linker flags (platform-dependent).
if platform.system() == 'Windows':
    _omp_compile = ['/openmp', '/O2']
    _omp_link    = []
else:
    _omp_compile = ['-fopenmp', '-O3']
    _omp_link    = ['-fopenmp']

extensions = [
    Extension(
        "mathlib._find_perimeter_cy",
        ["mathlib/_find_perimeter_cy.pyx"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mathlib._fmm3d_cy",
        ["mathlib/_fmm3d_cy.pyx"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=_omp_compile,
        extra_link_args=_omp_link,
    ),
    Extension(
        "mathlib._march_cy",
        ["mathlib/_march_cy.pyx"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=_omp_compile,
        extra_link_args=_omp_link,
    ),
    Extension(
        "mathlib._voxelize_cy",
        ["mathlib/_voxelize_cy.pyx"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=_omp_compile,
        extra_link_args=_omp_link,
    ),
]

setup(
    name='openMotor',
    version=appVersionStr,
    license='GPLv3',
    ext_modules=cythonize(extensions,
            nthreads=multiprocessing.cpu_count(),
            compiler_directives={'language_level': 3}
            ),
    packages=find_packages(),
    url='https://github.com/reilleya/openMotor',
    description='An open-source internal ballistics simulator for rocket motor experimenters',
    long_description=open('README.md').read(),
    cmdclass=cmdclass
)
