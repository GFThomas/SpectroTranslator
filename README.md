# SpectroTranslator

SpectroTranslator is a data-driven algorithm that transform spectroscopic parameters obtain from one survey into the base of another one.
The different translated catalogues, as well as the training/validation samples are available on *https://research.iac.es/proyecto/spectrotranslator/*

More information available on Thomas et al. 2024 (XXX)

## Updated February 20th 2024 by GuillaumeThomas *guillaume.thomas.astro .at. gmail.com*
-------------------------------------------------------------------------------------------------------------------------------



### REQUIREMENTS

You need Python3 installed on your computer. When you install the pipeline (see below), this will install the python package you need to run it.




### INSTALLATION
To do the installation, just run:

> ./install.sh




### Content
This repository contains:
- SpectroTranslator.py Main python code containing the core of the SpectroTranslator
- train_intrinsic.ipynb Jupyter notebook showing an example to train the *intrinsic* network of the SpectroTranslator
- train_extrinsic.ipynb Jupyter notebook showing an example to train the *extrinsic* network of the SpectroTranslator

The data to train the SpectroTranslator are available on  *https://research.iac.es/proyecto/spectrotranslator/*
