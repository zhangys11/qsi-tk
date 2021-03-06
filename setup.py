# from distutils.core import setup
from setuptools import find_packages, setup
import pathlib

setup(
    # Application name:
    name="qsi-tk",
    
    # Version number (initial):
    version="0.1.9",
    
    # Application author details:
    author="Yinsheng Zhang (Ph.D.)",
    author_email="oo@zju.edu.cn",
    
    # Packages
    packages=["qsi","qsi.fs","qsi.dr","qsi.cla","qsi.vis","qsi.io"],    

    # package_dir={'': 'qsi'},
    # package_dir={'qsi.dr': 'src/qsi/dr', 'qsi.cla': 'src/qsi/cla', 'qsi.vis': 'src/qsi/vis'},

    # Include additional files into the package
    include_package_data=True,
    
    # Details
    url="http://pypi.python.org/pypi/qsi_tk/",
    
    #
    license="LICENSE.txt",
    description="Data science toolkit (TK) from Quality-Safety research Institute (QSI).",
    
    long_description_content_type='text/markdown',
    long_description= open('README.md').read(),

    # Dependent packages (distributions)
    install_requires=[
        "flask",
        "pyDRMetrics",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "PyWavelets",
        "statsmodels",
        "pyNNRW",
        "pyCLAMs",
        "wCLAMs",
        "pyDRMetrics",
        "wDRMetrics",
        "cs1",
        #"ctgan", "torch"
    ],

    package_data={
        "": ["*.txt", "*.csv", "*.png", "*.jpg"],
    }
)

# To Build and Publish (for developer only), 
# Run: python setup.py sdist bdist_wheel; twine upload dist/*