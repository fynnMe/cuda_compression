#!/bin/bash

# skim https://docs.python.org/3/library/venv.html

# execute this file or the following lines to get into
#   a python virtual environment;
#   install dependencies via `pip`;

python -m venv virtualenvironment_python
source ./virtualenvironment_python/bin/activate

pip install pandas seaborn matplotlib
