import setuptools
from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setup(
    name="imageNet_protocols",
    version="0.0.1",
    author="Mike Suter",
    author_email="mike.suter@uzh.ch",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['access_files/**']},
)
