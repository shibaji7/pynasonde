python setup.py clean --all
rm -rf build *.egg-info
rm -rf `find -type d -name '.ipynb_checkpoints'`
rm -rf `find -type d -name '__pycache__'`
isort -rc -sl .
autoflake --remove-all-unused-imports -i -r .
isort -rc -m 3 .
black .
pip install .
ls $(python -c "import pynasonde; print(pynasonde.__path__[0])")