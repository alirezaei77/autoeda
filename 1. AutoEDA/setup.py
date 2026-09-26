from setuptools import find_packages, setup

setup(
    name="autoeda",
    version="0.1.0",
    description="Automated Exploratory Data Analysis toolkit.",
    author="Your Name",
    packages=find_packages(exclude=("tests", "examples")),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "missingno>=0.5.2",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "jinja2>=3.1.0",
    ],
    entry_points={"console_scripts": ["autoeda=autoeda.cli:main"]},
)