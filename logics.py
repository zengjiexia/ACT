import os
import re
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import imagej


class SimPullAnalysis:

    def __init__(self, data_path):
        self.error = 1 # When this value is 1, no error was detected in the object.
        self.path_program = os.path.dirname(__file__)
        self.path_data_main = data_path

        # Construct dirs for results
        self.path_result_main = data_path + '_results'
        if os.path.isdir(self.path_result_main) != 1:
            os.mkdir(self.path_result_main)
        self.path_result_raw = os.path.join(self.path_result_main, 'raw')
        if os.path.isdir(self.path_result_raw) != 1:
            os.mkdir(self.path_result_raw)
        self.path_result_samples = os.path.join(self.path_result_main, 'samples')
        if os.path.isdir(self.path_result_samples) != 1:
            os.mkdir(self.path_result_samples)
        naming_system = self.gather_project_info()
        if naming_system == 0:
            self.error = 'Invalid naming system for images. Currently supported naming systems are: XnYnRnWnCn, XnYnRnWn and Posn.'

    def gather_project_info(self):
        self.fov_paths = {} # dict - FoV name: path to the corresponding image
        for root, dirs, files in os.walk(self.path_data_main):
            for file in files:
                if file.endswith(".tif"):
                    try:
                        pos = re.findall(r"X\dY\dR\dW\dC\d", file)[-1]
                        naming_system = 'XnYnRnWnCn'
                    except IndexError:
                        try:
                            pos = re.findall(r'X\dY\dR\dW\d', file)[-1]
                            naming_system = 'XnYnRnWn'
                        except IndexError:
                            try:
                                pos = re.findall(r'Pos\d', file)[-1]
                                naming_system = 'Posn'
                            except IndexError:
                                return 0
                    self.fov_paths[pos] = os.path.join(root, file)
        self.wells = {} # dict - well name: list of FoV taken in the well
        try:
            list(self.fov_paths.keys())[0][7] # Check if the naming system is in 'pos\d'. IndexError would be raised if so.
            pass
        except IndexError:
            for fov in self.fov_paths:
                self.wells[fov] = [fov]
            return naming_system

        for fov in self.fov_paths:
            if fov[:4] in self.wells:
                self.wells[fov[:4]] += [fov]
            else:
                self.wells[fov[:4]] = [fov]
        return naming_system


    def call_ComDet(self, size, threshold, progress_signal=None):
        path_fiji = os.path.join(self.path_program, 'Fiji.app')
        IJ = imagej.init(path_fiji, headless=False)
        if progress_signal == None: #i.e. running in non-GUI mode
            IJ.ui().showUI()
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 1 # progress indicator
        for field in workload:
            imgFile = self.fov_paths[field]
            saveto = os.path.join(self.path_result_raw, field)
            saveto = saveto.replace("\\", "/")
            img = IJ.io().open(imgFile)
            IJ.ui().show(field, img)
            macro = """
            run("Z Project...", "projection=[Average Intensity]");
            run("Detect Particles", "ch1i ch1a="""+str(size)+""" ch1s="""+str(threshold)+""" rois=Ovals add=Nothing summary=Reset");
            selectWindow('Results');
            saveAs("Results", \""""+saveto+"""_results.csv\");
            close("Results");
            selectWindow('Summary');
            saveAs("Results", \""""+saveto+"""_summary.txt\");
            close(\""""+field+"""_summary.txt\");
            selectWindow(\"AVG_"""+field+"""\");
            saveAs("tif", \""""+saveto+""".tif\");
            close();
            close();
            """
            IJ.py.run_macro(macro)
            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1

        if progress_signal == None:
            IJ.py.run_macro("""run("Quit")""")
        else:
            IJ.getContext().dispose()
        return 1


    def generate_reports(self):
        # Generate sample reports
        for well in tqdm(self.wells):
            well_result = pd.DataFrame()
            for fov in self.wells[well]:
                try:
                    df = pd.read_csv(self.path_result_raw + '/' + fov + '_results.csv')
                    df = df.drop(columns=[' ', 'Channel', 'Slice', 'Frame'])
                    df['Abs_frame'] = fov[4:]
                    df['IntPerArea'] = df.IntegratedInt / df.NArea
                    well_result = pd.concat([well_result, df])
                except pd.errors.EmptyDataError:
                    pass
            well_result.to_csv(self.path_result_samples + '/' + well + '.csv', index=False)

        # Generate summary report
        summary_report = pd.DataFrame()
        for well in tqdm(self.wells):
            try:
                df = pd.read_csv(self.path_result_samples + '/' + well + '.csv')
                df_sum = pd.DataFrame.from_dict({
                    'Well': [well],
                    'NoOfFoV': [len(wells[well])],
                    'ParticlePerFoV': [len(df.index) / len(wells[well])],
                    'MeanSize': [df.NArea.mean()],
                    'MeanIntegrInt': [df.IntegratedInt.mean()],
                    'MeanIntPerArea': [df.IntPerArea.mean()]
                })
                summary_report = pd.concat([summary_report, df_sum])
            except pd.errors.EmptyDataError:
                pass
        summary_report.to_csv(self.path_result_main + '/Summary.csv', index=False)

        # Generate quality control report
        QC_data = pd.DataFrame()
        for well in tqdm(self.wells):
            try:
                df = pd.read_csv(self.path_result_samples + '/' + well + '.csv')
                df['Well'] = well
                df = df[['Well','Abs_frame', 'NArea', 'IntegratedInt', 'IntPerArea']]
                QC_data = pd.concat([QC_data, df])
            except pd.errors.EmptyDataError:
                pass
        QC_data = QC_data.reset_index(drop=True)
        QC_data.to_csv(self.path_result_main + '/QC.csv', index=False)



if __name__ == "__main__":

    path = input('Please input the path for analysis:\n')
    if os.path.isdir(path) != True:
    	print('Please input valid directory for data.')
    	quit()
    project = SimPullAnalysis(path)
    print('Launching: ' + path)
    size = input('Please input the estimated size of particles(in pixels):\n')
    threshold = input('Please input the threshold to apply(in nSD):\n')
    print('Picking up particles in Fiji...')
    project.call_ComDet(size=size, threshold=threshold)
    print('Generating reports...')
    project.generate_reports()
    print('Done.')



