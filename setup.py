from setuptools import setup, find_packages

setup(
    name="mini-alphaevolve",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "jinja2>=3.0.0",
        "pytest>=7.0.0",
        "tqdm>=4.65.0",
        "flake8>=6.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.8",
) 