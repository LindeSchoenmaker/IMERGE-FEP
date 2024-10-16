from setuptools import find_packages, setup

setup(
    name="RGroupInterm",
    version="0.1.0",
    author="Linde Schoenmaker",
    author_email="schoenmakerl1@lacdr.leidenuniv.nl",
    description="A molecular intermediate generator for relative binding free energy calculations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LindeSchoenmaker/RGroupInterm",
    packages=find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pillow>=10.1.0",
        "rdkit==2023.09.1",
        "scipy>=1.14.1"
    ]
)