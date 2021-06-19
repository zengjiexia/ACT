UKIA - Ultimate Kit for Image Analysis
======================================

A platform for the rapid analysis of both diffraction-limited and single-molecule localisation images.


Applications
------------
- Diffraction-Limited Analysis (DFLSP)
	
	This program is intended for rapid analysis of diffraction-limited images, to distinguish the protein aggregates (or other targets) captured by the SiMPull surface. The analysis process can also be used for other diffraction-limited images.
	
	Two methods are available:
		
	1. ComDet - This method relies on [Fiji(is just imagej)](https://imagej.net/Fiji) and [ComDet](https://github.com/ekatrukha/ComDet) - a plugin written by Eugene Katrukha at the Utrecht University.
		
	2. Trevor - A pure python alternative for the ComDet. (Written by Yunzhao Wu)
	
	An advanced thresholding method, Orthogonal Analysis, is provided. It can help you to distinguish the actual particles detected from the backgroud noises by setting a threshold based on the 'intensity per area' distribution of the spots.


Requirements
------------

- Python 3 (tested with Python 3.6-3.9)
	- os
	- sys
	- re
	- [tqdm - 4.60.0](https://pypi.org/project/tqdm/)
	- [astropy - 4.0.2](https://www.astropy.org/)
	- [PIL - 8.0.1](https://pypi.org/project/Pillow/)
	- [scipy - 1.5.2](https://www.scipy.org/)
	- [numpy](https://numpy.org/)
	- [pandas - 0.25.3](https://pandas.pydata.org/)
	- [scikit-image - 0.17.2](https://scikit-image.org/)
	- [PySide 6 - 6.0.2](https://pypi.org/project/PySide6/)
	- [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph)(Included locally)
    - [pyimagej](https://github.com/imagej/pyimagej) (install with conda-forge)
    - Openjdk - 8

- [Fiji(is just imagej)](https://imagej.net/Fiji) 
	- [ComDet](https://github.com/ekatrukha/ComDet)


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
conda activate DLA_python3
python /path_to/main.py
```


Contributors
------------
Zengjie Xia (University of Cambridge)

Yunzhao Wu (University of Cambridge)


Copyright 2021 Zengjie Xia, at the University of Cambridge.
