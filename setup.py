from setuptools import setup, find_packages

setup(
    name="cognize",
    version="0.1.1",
    author="Pulikanti Sashi Bharadwaj",
    description="Symbolic cognition engine for projection modeling and rupture control.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/heraclitus0/cognize",
    project_urls={
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=["cognize"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas"
    ],
    python_requires=">=3.8",
)
