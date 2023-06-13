ACT - Aggregate Charaterization Toolkit
======================================

A platform for the rapid analysis of both diffraction-limited and single-molecule localisation images.


Applications
------------
- Diffraction-Limited Analysis
	
	This program is intended for the rapid analysis of diffraction-limited images generated under fluorescence microscope. It is capable of distinguishing and characterising the protein aggregates (or other targets) captured on surface. 
	
	Two methods are available:
		
	1. ComDet - This method relies on [Fiji(is just imagej)](https://imagej.net/Fiji) and [ComDet](https://github.com/ekatrukha/ComDet) - a plugin written by Eugene Katrukha at the Utrecht University.
		
	2. PyStar - A pure python alternative for the ComDet. (Written by Yunzhao Wu)
	
	An advanced thresholding method, Orthogonal Analysis, is provided. It can help you to distinguish the actual particles detected from the backgroud noises by setting a threshold based on the 'intensity per area' distribution of the spots.
	

- Super-resolution Image Analysis
	
	Super-resolution images are stack of images taken when the fluorescent indicator being constantly bound and dissociated from the targets captured on the surface. This feature provides two image reconstruction methods - [GDSC SMLM 1](https://gdsc-smlm.readthedocs.io/en/latest/) and [ThunderSTORM](https://zitmen.github.io/thunderstorm/) to track the fluorescent spot through the stack and reconstruct into a image with higher resolution than diffraction limit.

	After image reconstruction, drift correction and particle clustering methods are also provided in the program, allowing you to further characterise the targets detected.


- Liposome Assay Analysis

	This workflow is the improved version of [Calcium Influx Assay](https://github.com/zengjiexia/CalciumInfluxAssay). It is an automated analysis program for [Ultrasensitive Measurement of Ca2+ Influx into Lipid Vesicles Induced by Protein Aggregates](https://doi.org/10.1002/anie.201700966) developed at the Klenerman Group. 



Requirements
------------

Full list please see ACT_python3.yml.

- Python 3 (tested with Python 3.6-3.9)
	- os
	- sys
	- re
	- datatime
	- [opencv - 4.7.0.72](https://pypi.org/project/opencv-contrib-python/)
	- [tqdm - 4.65.0](https://anaconda.org/conda-forge/tqdm)
	- [astropy - 5.2.2](https://www.astropy.org/)
	- [Pillow - 9.5.0](https://anaconda.org/conda-forge/pillow)
	- [scipy - 1.10.1](https://www.scipy.org/)
	- [numpy](https://numpy.org/)
	- [math](https://docs.python.org/3/library/math.html)
	- [tifffile](https://anaconda.org/conda-forge/tifffile)
	- [pandas - 1.4.1](https://pandas.pydata.org/)
	- [scikit-image - 0.19.3](https://scikit-image.org/)
	- [scikit-learn - 1.2.2](https://scikit-learn.org/)
	- [PySide 6 - 6.5.1](https://anaconda.org/conda-forge/pyside6)
	- [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph)(Included locally)
	- [pathos](https://anaconda.org/conda-forge/pathos)
	- [psutil](https://anaconda.org/conda-forge/psutil)
    - [pyimagej - 1.0.2](https://github.com/imagej/pyimagej) (must install with conda-forge)
    - [skan - 0.11.0](https://github.com/jni/skan)
    - [maven - 3.6.3](https://anaconda.org/conda-forge/maven)
    - Openjdk - 8

- [Fiji(is just imagej)](https://imagej.net/Fiji) 
	- [ComDet](https://github.com/ekatrukha/ComDet)
	- [GDSC SMLM 1](https://gdsc-smlm.readthedocs.io/en/latest/)
	- [ThunderSTORM](https://zitmen.github.io/thunderstorm/)


Installation
------------
Please install Anaconda for environment management.
```sh
cd /path-to-folder/
setup_conda_env.bat
```


Usage
-----
*The program only supports Windows system for now*

Command line tool:
```sh
conda activate ACT_python3
python /path_to/main.py
```


Contributors
------------
Zengjie Xia (University of Cambridge)

Yunzhao Wu (University of Cambridge)


Copyright 2021 Zengjie Xia, at the University of Cambridge.
