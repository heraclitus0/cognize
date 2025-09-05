from pathlib import Path
from setuptools import setup, find_packages

README = Path("README.md").read_text(encoding="utf-8")

setup(
    name="cognize",
    version="0.1.8",
    author="Pulikanti Sashi Bharadwaj",
    author_email="bharadwajpulikanti11@gmail.com",
    description="Programmable cognition for Python systems.",
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
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["cognize", "cognize.*"], exclude=("tests*", "examples*", "docs*")),
    include_package_data=True,
    package_data={"cognize": ["py.typed"]},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=["numpy>=1.26"],
    extras_require={
        "viz": ["pandas>=2.2", "matplotlib>=3.8", "seaborn>=0.13"],
        "dev": ["pytest>=7", "ruff>=0.4", "mypy>=1.8", "black>=23.12.1", "build>=1.0.3", "twine>=4.0.2"],
        "all": [
            "pandas>=2.2", "matplotlib>=3.8", "seaborn>=0.13",
            "pytest>=7", "ruff>=0.4", "mypy>=1.8", "black>=23.12.1", "build>=1.0.3", "twine>=4.0.2",
        ],
    },
)
