from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="algoq",
    version="0.1.0",
    packages=find_packages(where="algoq"),
    package_dir={"": "algoq"},
    install_requires=requirements,
    python_requires=">=3.8, <3.11",
    author="Hope Alemayehu",
    author_email="hopesp444@gmail.com",
    description="Reusable building blocks for quantum algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hope-Alemayehu/AlgoQ.git",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="quantum algorithms qaoa qubo qiskit pennylane",
)
