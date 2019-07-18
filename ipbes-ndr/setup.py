"""setup.py module for ipbes_ndr_analysis module."""
from Cython.Build import cythonize
import numpy
from setuptools.extension import Extension
from setuptools import setup

setup(
    name='ipbes ndr analysis',
    packages=[
        'ipbes_ndr_analysis',
    ],
    package_dir={
        'ipbes_ndr_analysis': 'src/ipbes_ndr_analysis'
    },

    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'node-and-date'},
    setup_requires=['setuptools_scm', 'cython', 'numpy'],
    include_package_data=True,
    ext_modules=cythonize(
        [Extension(
            "ipbes_ndr_analysis_cython",
            sources=["src/ipbes_ndr_analysis/ipbes_ndr_analysis_cython.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++",
        )],
        )
)
