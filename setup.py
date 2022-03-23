"""Setup file for the project."""
from setuptools import setup

from src import __docs__, __homepage__, __name__, __version__

setup(
    name=__name__,
    version=__version__,
    description=__docs__,
    url=__homepage__,
    packages=['src'],
    test_suite="tests",
    python_requires='>=3.6'
)
