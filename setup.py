from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cognize",
    version="0.1.3",
    author="Pulikanti Sashi Bharadwaj",
    description="Symbolic cognition engine for epistemic drift, rupture detection, and realignment.",
    long_description=long_description,
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
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["cognize", "cognize.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn"
    ],
    python_requires=">=3.8",
)
