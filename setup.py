from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return req_file.read().splitlines()


setup(
    name="deep_sort_reid",  # Your project name
    version="0.1.6",  # Your project version
    description="A re-mastered version of the original Deep Sort implementation, with added functionalities such as re-identification.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # If you have README.md
    url="https://github.com/cajmorgan/deep_sort_reid",  # URL of your project repo
    author="Caj Morgan",
    author_email="caj@maiosolutions.com",
    license="GPL-3.0",  # License type
    packages=find_packages(),  # Automatically find and include your packages
    install_requires=[  # Dependencies (from your requirements.txt)
        *parse_requirements("requirements.txt")
    ],
    python_requires=">=3.10.0",  # Minimum Python version
)
