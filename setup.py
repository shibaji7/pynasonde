# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pynasonde",
    version="0.0.2",
    packages=find_packages(),
    package_dir={"pynasonde": "pynasonde"},
    package_data={
        "pynasonde": ["config.toml", "digisonde_station_codes.csv"],
    },
    data_files=[
        (
            "pynasonde",
            ["pynasonde/config.toml", "pynasonde/digisonde_station_codes.csv"],
        )
    ],
    include_package_data=True,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    author="Shibaji Chakraborty",
    author_email="chakras4@erau.edu",
    maintainer="Shibaji Chakraborty",
    maintainer_email="chakras4@erau.edu",
    license="MIT",
    license_files=["LICENSE"],
    description=long_description,
    long_description=long_description,
    install_requires=[],
    keywords=["python", "ionosonde"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    url="https://github.com/shibaji7/pynasonde",
)
