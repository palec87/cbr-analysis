import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requires = [
    'numpy==1.18.5',
    'h5py',
    'pathlib',
    'matplotlib==3.2',
    'pandas',
    'scipy',
]

setuptools.setup(
    name="cbr-analysis",
    version="0.1.2",
    author="DP",
    author_email="david@stanka.de",
    description="OE analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/palec87/cbr-analysis/archive/v0.1.2.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requires,
)
