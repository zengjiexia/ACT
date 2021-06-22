import os
import re
import pandas as pd
from tqdm import tqdm
import imagej
from skimage import io
from skimage.morphology import disk, erosion, dilation, white_tophat, reconstruction
from skimage.measure import label, regionprops_table
import numpy as np
from astropy.convolution import RickerWavelet2DKernel
from PIL import Image
from scipy.stats import norm

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


    def call_ComDet(self, size, threshold, progress_signal=None, IJ=None):

        if progress_signal == None: #i.e. running in non-GUI mode
            path_fiji = os.path.join(self.path_program, 'Fiji.app')
            IJ = imagej.init(path_fiji, headless=False)
            IJ.ui().showUI()
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 1 # progress indicator

        #Check if the images are stack, and choose correct macro
        test_img = io.imread(list(self.fov_paths.values())[0])
        if len(test_img.shape) == 3: 
            stacked = True
        else:
            stacked = False

        for field in workload:
            imgFile = self.fov_paths[field]
            saveto = os.path.join(self.path_result_raw, field)
            saveto = saveto.replace("\\", "/")
            img = IJ.io().open(imgFile)
            IJ.ui().show(field, img)

            if stacked:
                macro = """
                run("Z Project...", "projection=[Average Intensity]");
                run("Detect Particles", "ch1i ch1a="""+str(size)+""" ch1s="""+str(threshold)+""" rois=Ovals add=Nothing summary=Reset");
                selectWindow('Results');
                saveAs("Results", \""""+saveto+"""_results.csv\");
                close("Results");
                close("Summary");
                selectWindow(\"AVG_"""+field+"""\");
                saveAs("tif", \""""+saveto+""".tif\");
                close();
                close();
                """
                IJ.py.run_macro(macro)
            else:
                macro = """
                run("Detect Particles", "ch1i ch1a="""+str(size)+""" ch1s="""+str(threshold)+""" rois=Ovals add=Nothing summary=Reset");
                selectWindow('Results');
                saveAs("Results", \""""+saveto+"""_results.csv\");
                close("Results");
                close("Summary");
                selectWindow(\""""+field+"""\");
                saveAs("tif", \""""+saveto+""".tif\");
                close();
                """
                IJ.py.run_macro(macro)

            # Remove edge particles
            df = pd.read_csv(saveto+'_results.csv')
            df = df.loc[(df['X_(px)'] >= 30) & (df['X_(px)'] <= 480)]
            df = df.loc[(df['Y_(px)'] >= 30) & (df['Y_(px)'] <= 480)]
            df = df.reset_index(drop=True)
            df.to_csv(saveto+'_results.csv')

            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1
            
        if progress_signal == None:
            IJ.py.run_macro("""run("Quit")""")
        else:
            IJ.py.run_macro("""
                if (isOpen("Log")) {
                 selectWindow("Log");
                 run("Close" );
                }
                """)

        return 1


    def call_Trevor(self, bg_thres = 1, tophat_disk_size=10, progress_signal=None, erode_size = 1):
        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 1 # progress indicator

        for field in workload:
            imgFile = self.fov_paths[field]
            saveto = os.path.join(self.path_result_raw, field)
            saveto = saveto.replace("\\", "/")
            img = io.imread(imgFile) # Read image
            img = img.astype(np.float64)
            if len(img.shape) == 3: # Determine if the image is a stack file with multiple slices
                img = np.mean(img, axis=0) # If true, average the image
            else:
                pass # If already averaged, go on processing

            img_size = np.shape(img)
            tophat_disk_size = 50
            tophat_disk = disk(tophat_disk_size) # create tophat structural element disk, diam = tophat_disk_size (typically set to 10)
            tophat_img = white_tophat(img, tophat_disk) # Filter image with tophat
            kernelsize = 1
            ricker_2d_kernel = RickerWavelet2DKernel(kernelsize)
            
            def convolve2D(image, kernel, padding=4, strides=1):
                
                # Cross Correlation
                kernel = np.flipud(np.fliplr(kernel))
            
                # Gather Shapes of Kernel + Image + Padding
                xKernShape = kernel.shape[0]
                yKernShape = kernel.shape[1]
                xImgShape = image.shape[0]
                yImgShape = image.shape[1]
            
                # Shape of Output Convolution
                xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
                yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
                output = np.zeros((xOutput, yOutput))
            
                # Apply Equal Padding to All Sides
                if padding != 0:
                    imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
                    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
                    #print(imagePadded)
                else:
                    imagePadded = image
            
                # Iterate through image
                for y in range(image.shape[1]):
                    # Exit Convolution
                    if y > image.shape[1] - yKernShape:
                        break
                    # Only Convolve if y has gone down by the specified Strides
                    if y % strides == 0:
                        for x in range(image.shape[0]):
                            # Go to next row once kernel is out of bounds
                            if x > image.shape[0] - xKernShape:
                                break
                            try:
                                # Only Convolve if x has moved by the specified Strides
                                if x % strides == 0:
                                    output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                            except:
                                break
            
                return output
            output = convolve2D(tophat_img, ricker_2d_kernel, padding=0)
            out_img = Image.fromarray(output)
            out_resize = out_img.resize(img_size)
            out_array = np.array(out_resize) 
            mu,sigma = norm.fit(out_array)
            threshold = mu + bg_thres*sigma
            out_array[out_array<threshold] = 0
            
            erode_img = erosion(out_array, disk(erode_size))
            dilate_img = dilation(erode_img, disk(erode_size))
            dilate_img[dilate_img>0] = 1
            mask = np.copy(dilate_img)
            
            io.imsave(saveto + '.tif', mask) # save masked image as result
            
            inverse_mask = 1-mask
            img_bgonly = inverse_mask*img
            seed_img = np.copy(img_bgonly) #https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html
            seed_img[1:-1, 1:-1] = img_bgonly.max()
            seed_mask = img_bgonly
            filled_img = reconstruction(seed_img, seed_mask, method='erosion')
            img_nobg = abs(img - filled_img)

            # Label the image to index all aggregates
            labeled_img = label(mask)
            # *save image

            intensity_list = []
            Abs_frame = []
            Channel = []
            Slice = []
            Frame = []

            # Get the number of particles
            num_aggregates = int(np.max(labeled_img))
            # Get profiles of labeled image
            df = regionprops_table(labeled_img, intensity_image=img, properties=['label', 'area', 'centroid', 'bbox'])
            df = pd.DataFrame(df)
            df.columns = [' ', 'NArea', 'X_(px)', 'Y_(px)', 'xMin', 'yMin', 'xMax', 'yMax']
            # Analyze each particle for integra
            for j in range(0, num_aggregates):
                current_aggregate = np.copy(labeled_img)
                current_aggregate[current_aggregate != j + 1] = 0
                current_aggregate[current_aggregate > 0] = 1
                intensity = np.sum(current_aggregate * img_nobg)
                intensity_list.append(intensity)
                
                Abs_frame.append(1)
                Channel.append(1)
                Slice.append(1)
                Frame.append(1)

            df['Abs_frame'] = Abs_frame
            df['Channel']= Channel
            df['Slice'] = Slice
            df['Frame'] = Frame
            df['IntegratedInt'] = intensity_list

            df.to_csv(saveto + '_results.csv', index=False) # save result.csv

            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1
        return 1


    def generate_reports(self, progress_signal=None):
        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.wells)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.wells)
            c = 1 # progress indicator
        # Generate sample reports
        for well in workload:
            well_result = pd.DataFrame()
            for fov in self.wells[well]:
                try:
                    df = pd.read_csv(self.path_result_raw + '/' + fov + '_results.csv')
                    df = df.drop(columns=[' ', 'Channel', 'Slice', 'Frame'])
                    df['FoV'] = fov
                    df['IntPerArea'] = df.IntegratedInt / df.NArea
                    well_result = pd.concat([well_result, df])
                except pd.errors.EmptyDataError:
                    pass
            well_result.to_csv(self.path_result_samples + '/' + well + '.csv', index=False)
            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1

        # Generate summary report
        summary_report = pd.DataFrame()
        for well in workload:
            try:
                df = pd.read_csv(self.path_result_samples + '/' + well + '.csv')
                df_sum = pd.DataFrame.from_dict({
                    'Well': [well],
                    'NoOfFoV': [len(self.wells[well])],
                    'ParticlePerFoV': [len(df.index) / len(self.wells[well])],
                    'MeanSize': [df.NArea.mean()],
                    'MeanIntegrInt': [df.IntegratedInt.mean()],
                    'MeanIntPerArea': [df.IntPerArea.mean()]
                })
                summary_report = pd.concat([summary_report, df_sum])
            except pd.errors.EmptyDataError:
                pass
            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1
        summary_report.to_csv(self.path_result_main + '/Summary.csv', index=False)

        # Generate quality control report
        QC_data = pd.DataFrame()
        for well in workload:
            try:
                df = pd.read_csv(self.path_result_samples + '/' + well + '.csv')
                df['Well'] = well
                df = df[['Well','FoV', 'NArea', 'IntegratedInt', 'IntPerArea']]
                QC_data = pd.concat([QC_data, df])
            except pd.errors.EmptyDataError:
                pass
            if progress_signal == None:
                pass
            else:
                progress_signal.emit(c)
                c += 1
        QC_data = QC_data.reset_index(drop=True)
        QC_data.to_csv(self.path_result_main + '/QC.csv', index=False)
        
        return 1


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



