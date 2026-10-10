from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()

setup(
    name="scprism",
    version="0.1.0",
    description="scPRISM: spatial graph-guided single-cell mapping",
    long_description=read("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="scPRISM Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "torch>=1.12.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "squidpy>=1.2.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
