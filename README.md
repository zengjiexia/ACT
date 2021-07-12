DiffractionLimitedAnalysis
===============

Rapid diffraction limited image analysis program.

Copyright 2021 Zengjie Xia, at the University of Cambridge

DiffractionLimitedAnalysis is intended for rapid analysis of diffraction limited images, to distinguish the protein aggregates (or other targets) captured by the SimPull surface. The image processing part of this code relies on [Fiji(is just imagej)](https://imagej.net/Fiji) and [ComDet](https://github.com/ekatrukha/ComDet) - a plugin written by Eugene Katrukha at the Utrecht University.

Requirements
------------

- Python 3 (tested with Python 3.6-3.9)
	- os
	- sys
	- re
	- [opencv - 4.1.2.30](https://pypi.org/project/opencv-contrib-python/)
	- [tqdm - 4.60.0](https://pypi.org/project/tqdm/)
	- [astropy - 4.0.2](https://www.astropy.org/)
	- [PIL - 8.0.1](https://pypi.org/project/Pillow/)
	- [scipy - 1.5.2](https://www.scipy.org/)
	- [numpy](https://numpy.org/)
	- [math](https://docs.python.org/3/library/math.html)
	- [tifffile](https://pypi.org/project/tifffile/)
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
*The program only support windows system for now*

Command line tool:
```sh
conda activate DLA_python3
python /path_to/main.py
```
