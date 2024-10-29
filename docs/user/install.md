<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
-->

# Installing Pynasonde 
---

[![License: MIT](https://img.shields.io/badge/License%3A-MIT-green)](https://choosealicense.com/licenses/mit/) 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) 
![GitHub Stable Release (latest by date)](https://img.shields.io/github/v/release/shibaji7/pynasonde)
[![Documentation Status](https://readthedocs.org/projects/pynasonde/badge/?version=latest)](https://pynasonde.readthedocs.io/en/latest/?badge=latest)


!!! Important 
    It is recommended to install pynasonde `pip`; however, please cite via the [DOI for the release]() 


## Prerequisites

Pynasonde requires **python 3.11** or later and **matplotlib 3.3.4** or later.

Depending on your operating system or distribution, the following package installers, development environments or data parsers are required: 
 
| Ubuntu      | OpenSuse       | Fedora        | OSX           | Windows       |
| ----------- | -------------- | ------------- | ------------- | ------------- |
| libyaml-dev | python3-PyYAML | libyaml-devel | Xcode/pip     | pip           |

You can check your python version using

`$ python --version` or 
`$ python3 --version`

!!! Note
    If you have already installed `pynasonde` you can use `pip3 install --upgrade pynasonde`

## Dependencies

pynasonde's setup will download the following dependencies: 

- [Git](https://git-scm.com/) (For developers)
- [pip3](https://help.dreamhost.com/hc/en-us/articles/115000699011-Using-pip3-to-install-Python3-modules)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [matplotlib 3.3.4+](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [pyproj](https://pyproj4.github.io/pyproj/stable/)
- [loguru](https://loguru.readthedocs.io/en/stable/)
- [SciencePlots](https://pypi.org/project/SciencePlots/1.0.2/)

!!! Note
    If you wish to plot coastlines or geographic projections you will need to install cartopy>=0.19 separately

### Cartopy 
[Cartopy](https://scitools.org.uk/cartopy/docs/latest/) is a Python package designed for geospatial data processing in order to produce maps and other geospatial data analyses. This library is used when invoking a projection system needing overlapped coastline maps in cable routs plots. 

For installing cartopy please follow the packages [installation](https://scitools.org.uk/cartopy/docs/latest/installing.html) instructions. For ubuntu here is good installation [link](https://techoverflow.net/2021/07/11/how-to-install-cartopy-on-ubuntu/) 

!!! Warning
    For cartopy to work with pynasonde please make sure it version `>=0.19`. Otherwise pynasonde may throw an exception if you try to use it.  


!!! Note
    cartopy can be a challenging package to install so please provide any information on troubleshooting or solutions to common issues on the [pynasonde GitHub](https://github.com/shibaji7/pynasonde) page. 



## Virtual Environments
It is recommended to install pynasonde in one of the suggested virtual environments if you have multiple python/pip 3 version on your computer, or do not want to affect the main system's python libraries. 

The following virtual environments have been tested by pynasonde developers:"

### pip Virtual Environment
Instructions can be found here [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

1. `$ python3 -m pip install --user virtualenv` (Install virtual environment package)
2. `$ python3 -m virtualenv <environment name>`  (Make your virtual environment)
3. `$ source <environment name>/bin/activate`  (Activate the virtual environment)
4. `$ pip install pynasonde`    (Install pynasonde)

!!! Note
    If you have multiple versions of python 3 on your machine, you can access a specific version by: `python<version number>`. 
    For example, if you want to install python 3.6 virtual environment: `python3.11 -m pip install --user virtualenv`.

### Anaconda Virtual Environment
Instructions can be found here [conda environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) and installing [anaconda](https://docs.anaconda.com/anaconda/install/)

1. `$ conda create -n yourenvname python=3.11 anaconda`
2. `$ conda activate yourenvname`
3. `$ pip install pynasonde`

#### Adding the environment to PyCharm

To set the project interpreter to the anaconda environment:

1. File -> Settings -> Project Folder -> Project Interpreter
2. Click the project Interpreter drop down list and click on show all.

    * If you don't see the environment you wish to use click the plus sign on the right side bar named "Add"
    * Select "Conda Environment" on the left side menu.
    * Click "Existing Environment" and give the interpreter field the path to your environment's python.exe and apply.

## Local Install
**pip3 install**

`pip3 install --user pynasonde`

## System Install 
`sudo pip3 install pynasonde`