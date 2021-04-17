from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, Signal, Slot
import pandas as pd
import sys
import imagej
import os

class LogTextEdit(QtWidgets.QPlainTextEdit):
    def write(self, message):
        if not hasattr(self, "flag"):
            self.flag = False
        message = message.replace('\r', '').rstrip()
        if message:
            method = "replace_last_line" if self.flag else "appendPlainText"
            QtCore.QMetaObject.invokeMethod(self,
                method,
                QtCore.Qt.QueuedConnection, 
                QtCore.Q_ARG(str, message))
            self.flag = True
        else:
            self.flag = False

    @QtCore.Slot(str)
    def replace_last_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.select(QtGui.QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertBlock()
        self.setTextCursor(cursor)
        self.insertPlainText(text)



class FijiWorker(QObject):
    finished = Signal()
    work_info = Signal(str, int)
    progress = Signal(int)

    def __init__(self, IJ, project, size, threshold):
        super().__init__()

        self.IJ = IJ
        self.project = project
        self.size = size
        self.threshold = threshold

    @QtCore.Slot()
    def run(self):

        def extract_FoV(path):
            """
            #get the name of field of views for a sample (format - XnYnRnWnCn)
            #para: path - string
            #return: fov_path - dict[fov] = path
            """
            fov_path = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.tif'):
                        fov_path[file[:10]] = os.path.join(root, file)
            return fov_path

        fov_paths = extract_FoV(self.project.path_data_main)

        self.work_info.emit('Locating particles...', len(fov_paths))

        for c, field in enumerate(sorted(fov_paths)):
            imgFile = fov_paths[field]
            saveto = os.path.join(self.project.path_result_raw, field)
            saveto = saveto.replace("\\", "/")
            img = self.IJ.io().open(imgFile)
            self.IJ.ui().show(field, img)
            macro = """
            run("Z Project...", "projection=[Average Intensity]");
            run("Detect Particles", "ch1i ch1a="""+str(self.size)+""" ch1s="""+str(self.threshold)+""" rois=Ovals add=Nothing summary=Reset");
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
            try:
                self.IJ.py.run_macro(macro)
            except:
                print(sys.exc_info())
            self.progress.emit(c+1)

        self.finished.emit()


class ReportWriter(QObject):
    finished = Signal()
    work_info = Signal(str, int)
    progress = Signal(int)

    def __init__(self, project):
        super().__init__()
        self.project = project


    @QtCore.Slot()
    def run(self):

        fovs, wells = self.project.gather_project_info()
        self.work_info.emit('Generating reports...', 3*len(wells))
        c = 1
        # Generate sample reports
        for well in wells:
            well_result = pd.DataFrame()
            for fov in wells[well]:
                try:
                    df = pd.read_csv(self.project.path_result_raw + '/' + fov + '_results.csv')
                    df = df.drop(columns=[' ', 'Channel', 'Slice', 'Frame'])
                    df['Abs_frame'] = fov[4:]
                    df['IntPerArea'] = df.IntegratedInt / df.NArea
                    well_result = pd.concat([well_result, df])
                except pd.errors.EmptyDataError:
                    pass
                self.progress.emit(c)
                c += 1
            well_result.to_csv(self.project.path_result_samples + '/' + well + '.csv', index=False)

        # Generate summary report
        summary_report = pd.DataFrame()
        for well in wells:
            try:
                df = pd.read_csv(self.project.path_result_samples + '/' + well + '.csv')
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
            self.progress.emit(c)
            c += 1
        summary_report.to_csv(self.project.path_result_main + '/Summary.csv', index=False)

        # Generate quality control report
        QC_data = pd.DataFrame()
        for well in wells:
            try:
                df = pd.read_csv(self.project.path_result_samples + '/' + well + '.csv')
                df['Well'] = well
                df = df[['Well','Abs_frame', 'NArea', 'IntegratedInt', 'IntPerArea']]
                QC_data = pd.concat([QC_data, df])
            except pd.errors.EmptyDataError:
                pass
            self.progress.emit(c)
            c += 1
        QC_data = QC_data.reset_index(drop=True)
        QC_data.to_csv(self.project.path_result_main + '/QC.csv', index=False)

        self.finished.emit()