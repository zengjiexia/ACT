#@String datapath
#@String resultpath
#@String size
#@String SD

from __future__ import with_statement
import os
from ij import IJ
from ij import WindowManager as wm


if __name__ == '__main__':

	def extract_FoV(path):
	"""
		#get the name of field of views for a sample (format - XnYnRnWnCn)
		#para: path - string
		#return: fov_path - dict[fov] = path
	"""
		fov_path = dict{}
		for root, dirs, files in os.walk(path):
			for file in files:
				if file.endswith('.tif'):
					fov_path[file[:10]] = os.path.join(root, file)
		return fov_path

	fov_paths = extract_FoV(datapath)

	for field in fov_paths:
		imgFile = fov_paths[field]

		IJ.open(imgFile)
		IJ.run("Rename..." , "title="+field)
		IJ.run("Z Project...", "projection=[Average Intensity]")

		IJ.run("Detect Particles", "ch1i ch1a="+size+" ch1s="+SD+" rois=Ovals add=Nothing summary=Reset")
		IJ.selectWindow('Results')
		IJ.saveAs('table', resultpath+'_results.csv')
		IJ.selectWindow('Summary')
		IJ.saveAs('text', resultpath+'_summary.txt')
		IJ.selectWindow('AVG_' + field)
		IJ.saveAs('tif', resultpath+'.tif')

	IJ.run("Quit")

