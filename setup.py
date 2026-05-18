from setuptools import setup, find_packages

setup(
    name="qau-qvs",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pytest>=8.0.0",
    ],
    author="ethcocoder",
    description="Quantum Absolute Unit (QAU) - Sovereign Substrate",
)
