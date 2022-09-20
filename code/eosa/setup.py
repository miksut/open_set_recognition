import setuptools
from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name="eosa",
    version="0.0.1",
    author="Mike Suter",
    author_email="mike.suter@uzh.ch",
    install_requires=requirements,
    packages=setuptools.find_packages(),
)
