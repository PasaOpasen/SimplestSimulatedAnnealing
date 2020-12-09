
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="SimplestSimulatedAnnealing", 
    version="1.0.0",
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer = ['Demetry Pascal'],
    description="An easy implementation of Hill Climbing algorithm for tasks with discrete variables in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/SimplestSimulatedAnnealing",
    keywords=['solve', 'optimization', 'problem', 'fast', 'combinatorial', 'easy', 'evolution', 'continual', 'simulated-annealing'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['numpy']
    
    )





