from setuptools import setup, find_packages

setup(
    name="p01_text_classification",
    version="0.1.0",
    description="Production NLP text classification pipeline — SMS Spam + SST-2",
    author="Your Name",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[],   # deps are in requirements.txt — keep setup.py lean
)
