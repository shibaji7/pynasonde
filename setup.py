# read the contents of your README file
from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pynasonde",
    version="0.1",
    packages=[
        "pynasonde",
        "pynasonde/riq/",
        "pynasonde/ngi/",
        "pynasonde/riq/headers/",
    ],
    package_dir={"pynasonde": "pynasonde"},
    package_data={"pynasonde": []},
    author="Shibaji Chakraborty",
    author_email="chakras4@erau.edu",
    maintainer="Shibaji Chakraborty",
    maintainer_email="chakras4@erau.edu",
    license="GNU GPL License",
    description=long_description,
    long_description=long_description,
    install_requires=[],
    keywords=["python", "ionosonde", "dynasonde"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/shibaji7/pynasonde",
)
