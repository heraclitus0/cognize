from pathlib import Path
from setuptools import setup, find_packages

README = Path("README.md").read_text(encoding="utf-8")

setup(
    name="cognize",
    version="0.1.6", 
    author="Pulikanti Sashi Bharadwaj",
    description="Programmable cognition for systems.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/heraclitus0/cognize",
    project_urls={
        "Documentation": "https://github.com/heraclitus0/cognize/blob/main/docs/USER_GUIDE.md",
        "Source": "https://github.com/heraclitus0/cognize",
        "Bug Tracker": "https://github.com/heraclitus0/cognize/issues",
    },
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["cognize", "cognize.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "viz": ["pandas>=2.0", "matplotlib>=3.6", "seaborn>=0.12"],
    },
)
