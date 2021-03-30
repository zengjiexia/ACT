import os
import subprocess
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imagej

class SimPullAnalysis:

    def __init__(self, data_path):
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


    def gather_project_info(self):
        fovs = {} # dict - FoV name: path to the corresponding image
        for root, dirs, files in os.walk(self.path_data_main):
            for file in files:
                if file.endswith(".tif"):
                    pos = re.findall(r"X\dY\dR\dW\dC\d", file)[-1]
                    fovs[pos] = os.path.join(root, file)
        wells = {} # dict - well name: list of FoV taken in the well
        for fov in fovs:
            if fov[:4] in wells:
                wells[fov[:4]] += [fov]
            else:
                wells[fov[:4]] = [fov]
        return fovs, wells


    def call_ComDet(self, size, SD):
        # *Use path_result_raw
        path_fiji = os.path.join(self.path_program, 'Fiji.app/ImageJ-win64.exe')
        path_macro = os.path.join(self.path_program, 'fiji_macro_v2.py')

        try:
            subprocess.call([path_fiji, "--ij2", "--console", "--run", path_macro, 'datapath=\'' + self.path_data_main + '\',' + 'resultpath=\'' + self.path_result_raw + '\',' + 'size=\'' + str(size) + '\',' + 'SD=\'' + str(SD) + '\''])
            print('Calculation completed without error.')
        except FileNotFoundError:
            print('Fiji path is incorrect.')


    def generate_reports(self):
        fovs, wells = self.gather_project_info()
        # Generate sample reports
        for well in wells:
            well_result = pd.DataFrame()
            for fov in wells[well]:
                try:
                    df = pd.read_csv(self.path_result_raw + fov + '_results.csv')
                    df = df.drop(columns=[' ', 'Channel', 'Slice', 'Frame'])
                    df['Abs_frame'] = fov[4:]
                    df['IntPerArea'] = df.IntegratedInt / df.NArea
                    well_result = pd.concat([well_result, df])
                except pd.errors.EmptyDataError:
                    pass
            well_result.to_csv(self.path_result_samples + well + '.csv', index=False)

        # Generate summary report
        summary_report = pd.DataFrame()
        for well in wells:
            df = pd.read_csv(self.path_result_samples + well + '.csv')
            df_sum = pd.DataFrame.from_dict({
                'Well': [well],
                'NoOfFoV': [len(wells[well])],
                'ParticlePerFoV': [len(df.index) / len(wells[well])],
                'MeanSize': [df.NArea.mean()],
                'MeanIntegrInt': [df.IntegratedInt.mean()],
                'MeanIntPerArea': [df.IntPerArea.mean()]
            })
            summary_report = pd.concat([summary_report, df_sum])
        summary_report.to_csv(self.path_result_main + '/Summary.csv', index=False)

        # Generate quality control report
        QC_data = pd.DataFrame()
        for well in wells:
            df = pd.read_csv(self.path_result_samples + well + '.csv')
            df['Well'] = well
            df = df[['Well','Abs_frame', 'NArea', 'IntegratedInt', 'IntPerArea']]
            QC_data = pd.concat([QC_data, df])
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
    project.call_ComDet_2(size=size, SD=threshold)
    print('Generating reports...')
    project.generate_reports()
    print('Done.')


