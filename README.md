DiffractionLimitedAnalysis
===============

Rapid diffraction limited image analysis program.

Copyright 2021 Zengjie Xia, Klenerman Group at the University of Cambridge

DiffractionLimitedAnalysis is intended for rapid analysis of diffraction limited images, to distinguish the protein aggregates (or other targets) captured by the SimPull surface. The image processing part of this code relies on [Fiji(is just imagej)](https://imagej.net/Fiji) and [ComDet](https://github.com/ekatrukha/ComDet) - a plugin written by Eugene Katrukha at the Utrecht University.

Requirements
------------

- Python 3 (tested with Python 3.6-3.9)
	- os
	- sys
	- re
	- pandas
	- seaborn
	- matplotlib
	- PySide 6
	- [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph)
    - pyimagej (install with conda-forge)

- [Fiji(is just imagej)](https://imagej.net/Fiji) 
	- [ComDet](https://github.com/ekatrukha/ComDet)

Usage
-----
*The program only support windows system for now*
Command line tool:
```sh
python /path_to/logics.py
```
Then follow the instructions.
