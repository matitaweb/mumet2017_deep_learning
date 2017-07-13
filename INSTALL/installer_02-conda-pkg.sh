#! /bin/bash

# Creating environment (sandbox instance called py3 [choose the name you want])
conda create -n py3 python=3 ipython
 
# Activating created environment
source activate py3

# Install package manager pip
conda install pip
# The installation installs the packages
 #pip install numpy
 #pip install pandas
 #pip install matplotlib
# which ipython is to be used in the environment? pip freeze shows it
pip freeze
# Installing ipython notebook
# conda install ipython-notebook

# Installing the packages
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
 
conda install -c conda-forge tensorflow