import setuptools
from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cbr_analysis",
    version="0.1.0",
    author="David",
    author_email="david@stanka.de",
    description="OE analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/palec87/cbr_analysis/archive/v0.1.0.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)