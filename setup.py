from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cognize",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A symbolic simulation tool for modeling projection drift, rupture, and adaptive recovery.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cognize",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces"
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "matplotlib"
    ],
)
