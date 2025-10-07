python setup.py clean --all
rm -rf dist site_build site junit.xml .coverage coverage.xml .pytest_cache .eggs 
rm -rf build *.egg-info
rm -rf `find -type d -name '.ipynb_checkpoints'`
rm -rf `find -type d -name '__pycache__'`
isort -rc -sl .
autoflake --in-place .
isort -rc -m 3 .
black .
pip install -e .[dev]

###################
# Install igrf and clean / build igrf from git 
# repo then use the repo to reshuffle to site-packages
###################