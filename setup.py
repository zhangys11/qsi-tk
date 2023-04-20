# from distutils.core import setup
from setuptools import setup

setup(
    # Application name:
    name="qsi-tk",

    # Version number:
    version="0.4.6",

    # Application author details:
    author="Yinsheng Zhang (Ph.D.)",
    author_email="oo@zju.edu.cn",

    # Packages
    packages=["qsi", "qsi.fs", "qsi.fs.glasso", "qsi.dr", "qsi.cla",
              "qsi.vis", "qsi.io", "qsi.data", "qsi.io.aug"],

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
    long_description=open('README.md', encoding='utf-8').read(),

    # Dependent packages (distributions)
    install_requires=[
        "flask",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "PyWavelets",
        "statsmodels",
        "h5py",
        "pyNNRW",
        "cla",
        "pyDRMetrics",
        "wDRMetrics",
        "pyMFDR",
        "cs1",
        "ctgan",  # "torch"
        "cvxpy",
        "asgl",
    ],

    package_data={
        "": ["*.txt", "*.csv", "*.png", "*.jpg", "*.json"],
    }
)

# To Build and Publish (for developer only),
# Run: python -m build
# Run: python -m pyc_wheel qsi_tk.whl  [optional]
# or
# Run: python setup.py sdist bdist_wheel; twine upload dist/*
