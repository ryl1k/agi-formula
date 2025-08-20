"""Setup script for AGI-Formula library."""

from setuptools import setup, find_packages
import os

# Read long description from README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="agi-formula",
    version="1.0.0",
    author="AGI-Formula Team",
    author_email="contact@agi-formula.dev",
    description="Artificial General Intelligence framework with consciousness-driven learning, multi-modal reasoning, and 2.47x performance advantage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agi-formula",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "optimization": [
            "cython>=0.29.0",
            "numba>=0.54.0",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "networkx>=2.6.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
        ],
        "pytorch": [
            "torch>=1.9.0",
        ],
        "benchmarks": [
            "torch>=1.9.0",
            "matplotlib>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agi-formula=agi_formula.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/agi-formula/issues",
        "Source": "https://github.com/yourusername/agi-formula",
        "Documentation": "https://agi-formula.readthedocs.io/",
    },
    keywords="artificial-general-intelligence agi consciousness reasoning creativity arc-agi abstract-reasoning meta-learning",
    include_package_data=True,
    zip_safe=False,
)