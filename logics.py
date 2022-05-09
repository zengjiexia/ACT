import os
import re
from datetime import datetime
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import math as ms
import tifffile as tiff
import imagej
from skimage import io
from skimage.morphology import disk, erosion, dilation, white_tophat, reconstruction
from skimage.measure import label, regionprops_table
from sklearn.cluster import DBSCAN
from astropy.convolution import RickerWavelet2DKernel
from PIL import Image
from scipy import ndimage
from scipy.stats import norm
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import psutil
import scyjava
import json
plugins_dir = os.path.join(os.path.dirname(__file__), 'Fiji.app/plugins')
scyjava.config.add_option(f'-Dplugins.dir={plugins_dir}')

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
            c = 0 # progress indicator

        # Check if the images are stack, and choose correct macro
        test_img = Image.open(list(self.fov_paths.values())[0])
        if test_img.n_frames > 1: 
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
            try:
                df = pd.read_csv(saveto+'_results.csv')
            except pd.errors.EmptyDataError:
               print('No spot found in FoV: ' + field)
            else:
                img_dimensions = Image.open(imgFile).size
                df = df.loc[(df['X_(px)'] >= img_dimensions[0] * 0.05) & (df['X_(px)'] <= img_dimensions[0] * 0.95)]
                df = df.loc[(df['Y_(px)'] >= img_dimensions[1] * 0.05) & (df['Y_(px)'] <= img_dimensions[1] * 0.95)]
                # Remove particles detected in the 5% pixels from the edges
                df = df.reset_index(drop=True)
                df.to_csv(saveto + '_results.csv')

            if progress_signal == None:
                pass
            else:
                c += 1
                progress_signal.emit(c) 

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

    
    def call_Trevor(self, bg_thres = 1, tophat_disk_size=50, progress_signal=None, erode_size = 1):
        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 0 # progress indicator
            
        num_cpu = multiprocessing.cpu_count()
        ram = psutil.virtual_memory().available
        estimated_cores = int(np.round(ram/1024/1024/1024/2))
        num_workers = np.minimum(num_cpu, estimated_cores)
        
        def process_img(img_index, fov_paths, path_result_raw, workload):
            
            field = workload[img_index]
            imgFile = fov_paths[field]
            saveto = os.path.join(path_result_raw, field)
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

            pad = np.zeros([img_size[0]+8, img_size[1]+8])
            pad[4:img_size[0]+4, 4:img_size[1]+4] = tophat_img
            pad_img = np.copy(pad)
            output = convolve2D(pad_img, ricker_2d_kernel, padding=0)
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
            mask[0:5, :] = 0
            mask[-5:, :] = 0
            mask[:, 0:5] = 0
            mask[:, -5:] = 0
            io.imsave(saveto + '.tif', mask) # save masked image as result
            
            inverse_mask = 1-mask
            img_bgonly = inverse_mask*img
            seed_img = np.copy(img_bgonly) # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html
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


        img_index = list(range(len(workload)))
        partial_func = partial(process_img, fov_paths=self.fov_paths, path_result_raw=self.path_result_raw, workload=workload)
        
        pool = Pool(num_workers)
        pool.map(partial_func, img_index)
        pool.close()
        pool.join()
  
        if progress_signal == None:
            pass
        else:
            c += 1
            progress_signal.emit(c)
        return 1


    def generate_reports(self, progress_signal=None):
        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.wells)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.wells)
            c = 0 # progress indicator
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
                c += 1
                progress_signal.emit(c)
                
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
                c += 1
                progress_signal.emit(c)

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
                c += 1
                progress_signal.emit(c)
                
        QC_data = QC_data.reset_index(drop=True)
        QC_data.to_csv(self.path_result_main + '/QC.csv', index=False)
        
        return 1



class LiposomeAssayAnalysis:

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
        self.gather_project_info()


    def gather_project_info(self):
        samples = [name for name in os.listdir(self.path_data_main) if not name.startswith('.') and name != 'results']
        if 'Ionomycin' in samples:
            self.samples = [self.path_data_main]
        else:
            self.samples = [os.path.join(self.path_data_main, sample) for sample in samples]

        ### Create result directory
        for sample in self.samples:
            if not os.path.isdir(sample.replace(self.path_data_main, self.path_result_raw)):
                os.mkdir((sample.replace(self.path_data_main, self.path_result_raw)))

    
    def run_analysis(self, threshold, progress_signal=None, log_signal=None):

        def extract_filename(path):
            """
            walk through a directory and put names of all tiff files into an ordered list
            para: path - string
            return: filenames - list of string 
            """

            filenames = []
            for root, dirs, files in os.walk(path):
                for name in files:
                    if name.endswith('.tif'):
                        filenames.append(name)
            filenames = sorted(filenames)
            return filenames


        def average_frame(path):
            """
            input 'path' for stacked tiff file and the 'number of images' contained
            separate individual images from a tiff stack.
            para: path - string
            return: ave_img - 2D array
            """

            ori_img = tiff.imread(path)
            ave_img = np.mean(ori_img, axis=0)
            ave_img = ave_img.astype('uint16')

            return ave_img


        def img_alignment(Ionomycin, Sample, Blank):
            """
            image alignment based on cross-correlation
            Ionomycin image is the reference image
            para: Ionomycin, Sample, Blank - 2D array
            return: Corrected_Sample, Corrected_Blank - 2D array
            """

            centre_ = (Ionomycin.shape[0]/2, Ionomycin.shape[1]/2)
            # 2d fourier transform of averaged images
            FIonomycin = np.fft.fft2(Ionomycin)
            FSample = np.fft.fft2(Sample)
            FBlank = np.fft.fft2(Blank)

            # Correlation based on Ionomycin image
            FRIS = FIonomycin*np.conj(FSample)
            FRIB = FIonomycin*np.conj(FBlank)

            RIS = np.fft.ifft2(FRIS)
            RIS = np.fft.fftshift(RIS)
            RIB = np.fft.ifft2(FRIB)
            RIB = np.fft.fftshift(RIB)

            [i, j] = np.where(RIS == RIS.max())
            [g, k] = np.where(RIB == RIB.max())

            # offset values
            IS_x_offset = i-centre_[1]
            IS_y_offset = j-centre_[0]
            IB_x_offset = g-centre_[1]
            IB_y_offset = k-centre_[0]

            # Correction
            MIS = np.float64([[1, 0, IS_y_offset], [0, 1, IS_x_offset]])
            Corrected_Sample = cv2.warpAffine(Sample, MIS, Ionomycin.shape)
            MIB = np.float64([[1, 0, IB_y_offset], [0, 1, IB_x_offset]])
            Corrected_Blank = cv2.warpAffine(Blank, MIB, Ionomycin.shape)

            return Corrected_Sample, Corrected_Blank


        def peak_locating(data, threshold):
            """
            Credit to Dr Daniel R Whiten
            para: data - 2D array
            para: threshold - integer
            return: xy_thresh - 2D array [[x1, y1], [x2, y2]...]
            """

            data_max = ndimage.filters.maximum_filter(data, 3)
            maxima = (data == data_max)
            data_min = ndimage.filters.minimum_filter(data, 3)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
            xy_thresh = np.zeros((0, 2))
            for row in xy:
                a = row[0]
                b = row[1]
                if (a > 30) and (a < 480) and (b > 30) and (b < 480):
                    ab = np.array([np.uint16(a), np.uint16(b)])
                    xy_thresh = np.vstack((xy_thresh, ab))
            xy_thresh = xy_thresh[1:] 

            return xy_thresh


        def intensities(image_array, peak_coor, radius=3):
            """
            When the local peak is found, extract all the coordinates of pixels in a 'radius'
            para: image_array - 2D array
            para: peak_coor - 2D array [[x1, y1], [x2, y2]]
            para: radius - integer
            return: intensities - 2D array [[I1], [I2]]
            """

            x_ind, y_ind = np.indices(image_array.shape)
            intensities = np.zeros((0,1))
            for (x, y) in peak_coor:
                intensity = 0
                circle_points = ((x_ind - x)**2 + (y_ind - y)**2) <= radius**2
                coor = np.where(circle_points == True)
                coor = np.array(list(zip(coor[0], coor[1])))
                for j in coor:
                    intensity += image_array[j[0], j[1]]
                intensities = np.vstack((intensities, intensity))

            return intensities


        def influx_qc(field, peaks, influx_df):
            ### Remove error measurements ###
            """ 
            if 100% < influx < 110% take as 100%
            if -10% < influx < 0% take as 0
            if influx calculated to be nan or <-10% or >110% count as error
            """ 
            influx_df['Influx'] = [100 if i >= 100 and i <= 110 else i for i in influx_df['Influx']]
            influx_df['Influx'] = [0 if i <= 0 and i >= -10 else i for i in influx_df['Influx']]
            influx_df['Influx'] = ['error' if ms.isnan(np.float(i)) or i < -10 or i > 110 else i for i in influx_df['Influx']]

            ### Generate a dataframe which contains the result of current FoV ###
            field_result = pd.concat([
                pd.DataFrame(np.repeat(field, len(peaks)), columns=['Field']),
                pd.DataFrame(peaks, columns=['X', 'Y']),
                influx_df
                ],axis = 1)

            ### Filter out error data ###
            field_result = field_result[field_result.Influx != 'error']
            
            ### Get field summary ###
            try:
                field_error = (influx_df.Influx == 'error').sum()
            except (AttributeError, FutureWarning) as e:
                field_error = 0

            field_summary = pd.DataFrame({
                "FoV": [field],
                "Mean influx": [field_result.Influx.mean()],
                "Total liposomes": [len(peaks)],
                "Valid liposomes": [len(peaks)-field_error],
                "Invalid liposomes": [field_error]
                })
            return field_result, field_summary


        def pass_log(text):
            if log_signal == None:
                print(text)
            else:
                log_signal.emit(text)


        def process_img(img_index, workload, threshold):

            sample = workload[img_index]
            sample_summary = pd.DataFrame()

            # report which sample is running to log window
            #pass_log('Running sample: ' + sample)
            
            ionomycin_path = os.path.join(sample, 'Ionomycin')
            sample_path = os.path.join(sample, 'Sample')
            blank_path = os.path.join(sample, 'Blank')

            #if not os.path.isdir(ionomycin_path):
                #pass_log('Skip ' + sample + '. No data found in the sample folder.')

            ### Obtain filenames for fields of view ###
            field_names = extract_filename(ionomycin_path)

            for field in field_names:
                ### Average tiff files ###
                ionomycin_mean = average_frame(os.path.join(ionomycin_path, field))
                sample_mean = average_frame(os.path.join(sample_path, field))
                blank_mean = average_frame(os.path.join(blank_path, field))
                
                ### Align blank and sample images to the ionomycin image ###
                sample_aligned, blank_aligned = img_alignment(ionomycin_mean, sample_mean, blank_mean)

                ### Locate the peaks on the ionomycin image ###
                peaks = peak_locating(ionomycin_mean, threshold)
                
                if len(peaks) == 0:
                    #pass_log('Field ' + field + ' of sample ' + sample +' ignored due to no liposome located in this FoV.')
                    field_summary = pd.DataFrame({
                        "FoV": [field],
                        "Mean influx": [0],
                        "Total liposomes": [0],
                        "Valid liposomes": [0],
                        "Invalid liposomes": [0]
                        })
                    sample_summary = pd.concat([sample_summary, field_summary])
                else:
                    ### Calculate the intensities of peaks with certain radius (in pixel) ###
                    ionomycin_intensity = intensities(ionomycin_mean, peaks)
                    sample_intensity = intensities(sample_aligned, peaks)
                    blank_intensity = intensities(blank_aligned, peaks)

                    ### Calculate influx of each single liposome and count errors ###
                    influx_df = pd.DataFrame((sample_intensity - blank_intensity)/(ionomycin_intensity - blank_intensity)*100, columns=['Influx'])

                    field_result, field_summary = influx_qc(field, peaks, influx_df)
                    field_result.to_csv(os.path.join(sample.replace(self.path_data_main, self.path_result_raw), field+".csv"))

                    sample_summary = pd.concat([sample_summary, field_summary])
            sample_summary.to_csv(sample.replace(self.path_data_main, self.path_result_raw) + ".csv")



        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.samples)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.samples)
            c = 0 # progress indicator

        num_cpu = multiprocessing.cpu_count()
        ram = psutil.virtual_memory().available
        estimated_cores = int(np.round(ram/1024/1024/1024/2))
        num_workers = np.minimum(num_cpu, estimated_cores)

        img_index = list(range(len(workload)))
        partial_func = partial(process_img, workload =workload, threshold=threshold)

        pool = Pool(num_workers)
        pool.map(partial_func, img_index)
        pool.close()
        pool.join()

        # Report progress
        if progress_signal != None:
            c += 1
            progress_signal.emit(c)

        return 1


    def generate_reports(self, progress_signal=None):
        workload = [f for f in os.listdir(self.path_result_raw) if os.path.isfile(os.path.join(self.path_result_raw, f))]
        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(workload)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(workload)
            c = 0 # progress indicator

        summary_df = pd.DataFrame()
        for file in workload:
            file_path = os.path.join(self.path_result_raw, file)
            df = pd.read_csv(file_path)
            df['Well'] = df['FoV'].str.findall(r"X\dY\d")
            df['Well'] = df['Well'].str.get(0)
            df = df.groupby('Well').agg({'Mean influx': 'mean',
                                        'Total liposomes': 'sum',
                                        'Valid liposomes': 'sum',
                                        'Invalid liposomes': 'sum'})
            df = df.reset_index(drop=False)
            df['Sample'] = file[:-4]
            summary_df = pd.concat([summary_df, df])

            if progress_signal != None:
                c += 1
                progress_signal.emit(c)

        cols = summary_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        summary_df = summary_df[cols]
        summary_df.to_csv(self.path_result_main + '/Summary.csv', index=False)
        
        return 1



class SuperResAnalysis:
    
    def __init__(self, data_path): 
        self.error = 1 # When this value is 1, no error was detected in the object.
        self.path_program = os.path.dirname(__file__)
        self.path_data_main = data_path
        self.gather_project_info()


    def update_parameters(self, parameters):
        self.parameters = parameters


    def gather_project_info(self):

        self.fov_paths = {} # dict - FoV name: path to the corresponding image

        for root, dirs, files in os.walk(self.path_data_main):
            for file in files:
                if file.endswith(".tif"):
                    try:
                        pos = re.findall(r"X\dY\dR\dW\dC\d", file)[-1]
                    except IndexError:
                        self.error = 'Error in the naming system of the images. Please make sure the image names contain coordinate in form of XnYnRnWnCn.'
                        return 0
                    self.fov_paths[pos] = os.path.join(root, file)

        self.wells = {} # dict - well name: list of FoV taken in the well

        for fov in self.fov_paths:
            if fov[:4] in self.wells:
                self.wells[fov[:4]] += [fov]
            else:
                self.wells[fov[:4]] = [fov]

        # Check if the images are stacks
        test_img = Image.open(list(self.fov_paths.values())[0])
        self.img_frames = test_img.n_frames
        print('Test image number of frames: ' + str(self.img_frames))
        if self.img_frames < 2: 
            self.error = 'The images are not stacked. Please check.'
            return 0

        # Get image dimensions
        self.dimensions = test_img.size
        print('Test image dimensions: ' + str(self.dimensions))

        return 1


    def _compose_fiji_macro(self, field_name):
        if self.parameters['method'] == 'GDSC SMLM 1':
            self.macro = """
            run("Peak Fit", "template=[None] config_file=["""+self.path_result_raw + '/gdsc.smlm.settings.xml' +"""] calibration="""+str(self.parameters['pixel_size'])+""" gain="""+str(self.parameters['camera_gain'])+""" exposure_time="""+str(self.parameters['exposure_time'])+""" initial_stddev0=2.000 initial_stddev1=2.000 initial_angle=0.000 smoothing=0.50 smoothing2=3 search_width=3 fit_solver=[Least Squares Estimator (LSE)] fit_function=Circular local_background camera_bias="""+str(self.parameters['camera_bias'])+""" fit_criteria=[Least-squared error] significant_digits=5 coord_delta=0.0001 lambda=10.0000 max_iterations=20 fail_limit=10 include_neighbours neighbour_height=0.30 residuals_threshold=1 duplicate_distance=0.50 shift_factor=2 signal_strength="""+str(self.parameters['signal_strength'])+""" width_factor=2 precision="""+str(self.parameters['precision'])+""" min_photons="""+str(self.parameters['min_photons'])+""" results_table=Uncalibrated image=[Localisations (width=precision)] weighted equalised image_precision=5 image_scale="""+str(self.parameters['scale'])+""" results_dir=["""+self.path_result_raw+"""] local_background camera_bias="""+str(self.parameters['camera_bias'])+""" fit_criteria=[Least-squared error] significant_digits=5 coord_delta=0.0001 lambda=10.0000 max_iterations=20 stack");
            selectWindow(\""""+field_name+""" (LSE) SuperRes\");
            saveAs("tif", \""""+self.path_result_raw + '/SR_' + field_name+""".tif\");
            close(\"SR_"""+field_name+""".tif\");
            selectWindow("Fit Results");
            saveAs("Results", \""""+self.path_result_raw + '/' + field_name+"""_results.csv\");
            close("Fit Results");
            close("Log");
            close(\""""+field_name+"""\");
            """
        elif self.parameters['method'] == 'ThunderSTORM':
            self.macro = """
            run("Camera setup", "offset="""+str(self.parameters['camera_bias'])+""" quantumefficiency="""+str(self.parameters['quantum_efficiency'])+""" isemgain=true photons2adu=3.6 gainem="""+str(self.parameters['camera_gain'])+""" pixelsize="""+str(self.parameters['pixel_size'])+"""");
            run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification="""+str(self.parameters['scale'])+""" colorize=false threed=false shifts=2 repaint="""+str(int(self.parameters['exposure_time']))+"""");
            run("Export results", "floatprecision=5 filepath="""+self.path_result_raw + '/' + field_name+"""_results.csv fileformat=[CSV (comma separated)] sigma=true intensity=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty_xy=true frame=true");
            selectWindow('Averaged shifted histograms');
            saveAs("tif", \""""+self.path_result_raw + '/SR_' + field_name+""".tif\");
            close(\"SR_"""+field_name+""".tif\");
            close(\""""+field_name+"""\");
            """
        

    def superRes_reconstruction(self, progress_signal=None, IJ=None):

        # Construct dirs for results
        self.timeStamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.path_result_main = (self.path_data_main + '_' + self.timeStamp + '_' + self.parameters['method'])
        self.path_result_main = self.path_result_main.replace("\\", "/") # fiji only reads path with /
        self.path_result_main = self.path_result_main.replace(" ", "_") # fiji only reads path with /
        if os.path.isdir(self.path_result_main) != 1:
            os.mkdir(self.path_result_main)
        self.path_result_raw = os.path.join(self.path_result_main, 'raw')
        self.path_result_raw = self.path_result_raw.replace("\\", "/")# fiji only reads path with /
        if os.path.isdir(self.path_result_raw) != 1:
            os.mkdir(self.path_result_raw)

        with open(os.path.join(self.path_result_main, 'parameters.txt'), 'w') as js_file:
            json.dump(self.parameters, js_file)

        if progress_signal == None: #i.e. running in non-GUI mode
            path_fiji = os.path.join(self.path_program, 'Fiji.app')
            IJ = imagej.init(path_fiji, headless=False)
            IJ.ui().showUI()
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 0 # progress indicator

        for field in workload:
            imgFile = self.fov_paths[field]
            #saveto = os.path.join(self.path_result_raw, field)
            #saveto = saveto.replace("\\", "/")
            img = IJ.io().open(imgFile)
            IJ.ui().show(field, img)

            self._compose_fiji_macro(field)
            IJ.py.run_macro(self.macro)

            if progress_signal == None:
                pass
            else:
                c += 1
                progress_signal.emit(c)
        return 1


    def _GDSC_TS_IOadapter(self, GDSC_result=None, TS_result=None):
        GDSC_df = pd.read_csv(GDSC_result) # Read file
        GDSC_df = GDSC_df.sort_values(by = ['Frame']) # sort by frame number

        if TS_result == None: # i.e. convert GSDC to TS
            TS_df = GDSC_df[['Frame', 'X', 'Y', 'origValue']] # take out important columns
            TS_df.rename(columns = {'Frame': 'frame', 'X': 'x [nm]', 'Y': 'y [nm]', 'origValue': 'intensity [photon]'}, inplace = True) # Replace column names into ThunderSTORM format
            TS_df['x [nm]'] = self.parameters['pixel_size'] * TS_df['x [nm]'] # Convert between pixel and nm
            TS_df['y [nm]'] = self.parameters['pixel_size'] * TS_df['y [nm]'] # Convert between pixel and nm
            TS_df.to_csv(GDSC_result.replace('.csv', '_TS.csv'), index = False) # Write ThunderSTORM file for fiducial correction with FIJI

        else: # i.e. Feed corrected X,Y back to GDSC file
            TS_df = pd.read_csv(TS_result) # Read file
            TS_df['x [nm]'] = TS_df['x [nm]'] / self.parameters['pixel_size'] # Convert between pixel and nm
            TS_df['y [nm]'] = TS_df['y [nm]'] / self.parameters['pixel_size'] # Convert between pixel and nm
            GDSC_corrected_df = GDSC_df.copy()
            GDSC_corrected_df['X'] = TS_df['x [nm]'].values # Replace values with corrected values
            GDSC_corrected_df['Y'] = TS_df['y [nm]'].values # Replace values with corrected values
            GDSC_corrected_df.to_csv(TS_result.replace('_corrected_TS.csv', '_corrected.csv'), index = False)


    def _fidCorr_TS_fiducialMarkers(self, field_name, IJ=None):
        if self.parameters['method'] == 'ThunderSTORM':
            self.macro = """
            run("Import results", "detectmeasurementprotocol=true filepath="""+self.path_result_raw+ "/" +field_name+"""_results.csv fileformat=[CSV (comma separated)] livepreview=true rawimagestack= startingframe=1 append=false");
            run("Show results table", "action=drift smoothingbandwidth=0.25 method=[Fiducial markers] ontimeratio="""+str(self.parameters['min_visibility'])+""" distancethr="""+str(self.parameters['max_distance'])+""" save=false");
            run("Export results", "floatprecision=5 filepath="""+self.path_result_fid+"/"+field_name+"""_corrected.csv fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty_xy=true frame=true");
            selectWindow("Averaged shifted histograms");
            saveAs("tif", \""""+self.path_result_fid+"/SR_"+field_name+"""_corrected.tif\");
            close(\"SR_"""+field_name+"""_corrected.tif\");
            selectWindow("Drift");
            saveAs("tif",\""""+self.path_result_fid+"/"+field_name+"""_drift.tif\");
            close(\""""+field_name+"""_drift.tif\");
            """
            IJ.py.run_macro(self.macro)

        elif self.parameters['method'] == 'GDSC SMLM 1':
            self._GDSC_TS_IOadapter(GDSC_result=self.path_result_raw+ "/" + field_name+"_results.csv") # Convert the GDSC result to TS format and save as _TS.csv
            self.macro = """
            run("Import results", "detectmeasurementprotocol=true filepath="""+self.path_result_raw+ "/" +field_name+"""_results_TS.csv fileformat=[CSV (comma separated)] livepreview=true rawimagestack= startingframe=1 append=false");
            run("Show results table", "action=drift smoothingbandwidth=0.25 method=[Fiducial markers] ontimeratio="""+str(self.parameters['min_visibility'])+""" distancethr="""+str(self.parameters['max_distance'])+""" save=false");
            run("Export results", "floatprecision=5 filepath="""+self.path_result_fid+"/"+field_name+"""_corrected_TS.csv fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty_xy=true frame=true");
            selectWindow("Averaged shifted histograms");
            saveAs("tif", \""""+self.path_result_fid+"/SR_"+field_name+"""_corrected.tif\");
            close(\"SR_"""+field_name+"""_corrected.tif\");
            selectWindow("Drift");
            saveAs("tif",\""""+self.path_result_fid+"/"+field_name+"""_drift.tif\");
            close(\""""+field_name+"""_drift.tif\");
            """
            IJ.py.run_macro(self.macro)

            self._GDSC_TS_IOadapter(GDSC_result=self.path_result_raw+ "/" + field_name+"_results.csv", TS_result=self.path_result_fid+ "/" + field_name+"_corrected_TS.csv") # Feed the corrected X, Y coordinates back to GDSC result file


    def _fidCorr_TS_crossCorrelation(self, field_name, IJ=None):
        if self.parameters['method'] == 'ThunderSTORM':
            self.macro = """
            run("Import results", "detectmeasurementprotocol=true filepath="""+self.path_result_raw+ "/" + field_name+"""_results.csv fileformat=[CSV (comma separated)] livepreview=true rawimagestack= startingframe=1 append=false");
            run("Show results table", "action=drift magnification="""+str(self.parameters['magnification'])+""" method=[Cross correlation] ccsmoothingbandwidth=0.25 save=false steps="""+str(self.parameters['bin_size'])+""" showcorrelations=false");
            run("Export results", "floatprecision=5 filepath="""+self.path_result_fid+"/"+field_name+"""_corrected.csv fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty_xy=true frame=true");
            selectWindow("Averaged shifted histograms");
            saveAs("tif", \""""+self.path_result_fid+"/SR_"+field_name+"""_corrected.tif\");
            close(\"SR_"""+field_name+"""_corrected.tif\");
            selectWindow("Drift");
            saveAs("tif",\""""+self.path_result_fid+"/"+field_name+"""_drift.tif\");
            close(\""""+field_name+"""_drift.tif\");
            """
            IJ.py.run_macro(self.macro)

        elif self.parameters['method'] == 'GDSC SMLM 1':
            self._GDSC_TS_IOadapter(GDSC_result=self.path_result_raw+ "/" + field_name+"_results.csv") # Convert the GDSC result to TS format and save as _TS.csv
            self.macro = """
            run("Import results", "detectmeasurementprotocol=true filepath="""+self.path_result_raw+ "/" + field_name+"""_results_TS.csv fileformat=[CSV (comma separated)] livepreview=true rawimagestack= startingframe=1 append=false");
            run("Show results table", "action=drift magnification="""+str(self.parameters['magnification'])+""" method=[Cross correlation] ccsmoothingbandwidth=0.25 save=false steps="""+str(self.parameters['bin_size'])+""" showcorrelations=false");
            run("Export results", "floatprecision=5 filepath="""+self.path_result_fid+"/"+field_name+"""_corrected_TS.csv fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty_xy=true frame=true");
            selectWindow("Averaged shifted histograms");
            saveAs("tif", \""""+self.path_result_fid+"/SR_"+field_name+"""_corrected.tif\");
            close(\"SR_"""+field_name+"""_corrected.tif\");
            selectWindow("Drift");
            saveAs("tif",\""""+self.path_result_fid+"/"+field_name+"""_drift.tif\");
            close(\""""+field_name+"""_drift.tif\");
            """
            IJ.py.run_macro(self.macro)

            self._GDSC_TS_IOadapter(GDSC_result=self.path_result_raw+ "/" + field_name+"_results.csv", TS_result=self.path_result_fid+ "/" + field_name+"_corrected_TS.csv") # Feed the corrected X, Y coordinates back to GDSC result file


    def _fidCorr_GDSC_autoFid(self, field_name):
        if self.parameters['method'] == 'GDSC SMLM 1':
            pass


    def superRes_fiducialCorrection(self, progress_signal=None, IJ=None):
        if progress_signal == None: #i.e. running in non-GUI mode
            path_fiji = os.path.join(self.path_program, 'Fiji.app')
            IJ = imagej.init(path_fiji, headless=False)
            IJ.ui().showUI()
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 0 # progress indicator

        for field in workload:
            
            if self.parameters['fid_method'] == 'Fiducial marker - ThunderSTORM':
                self.path_result_fid = self.path_result_main + "/ThunderSTORM_FidMarker_" + str(self.parameters['max_distance']) + "_" + str(self.parameters['min_visibility'])
                if os.path.isdir(self.path_result_fid) != 1:
                    os.mkdir(self.path_result_fid)

                self._fidCorr_TS_fiducialMarkers(field, IJ=IJ)
                

            elif self.parameters['fid_method'] == 'Cross-correlation - ThunderSTORM':
                self.path_result_fid = self.path_result_main + "/ThunderSTORM_CrossCorrelation_" + str(self.parameters['bin_size']) + "_" + str(self.parameters['magnification'])
                if os.path.isdir(self.path_result_fid) != 1:
                    os.mkdir(self.path_result_fid)

                self._fidCorr_TS_crossCorrelation(field, IJ=IJ)
                

            if progress_signal == None:
                pass
            else:
                c += 1
                progress_signal.emit(c)
        return 1


    def _cluster_dataFiltering(self, field_name):
        paras = self.parameters['filter']
        # The result files are named differently for drift corrected and uncorrected data
        if self.path_result_fid.endswith('raw'):
            file_dir = os.path.join(self.path_result_fid, field_name+'_results.csv')
        else:
            file_dir = os.path.join(self.path_result_fid, field_name+'_corrected.csv')
        df = pd.read_csv(file_dir)

        # Filter spots based on chosen parameters
        # For GDSC, sigma-SD, precision-Precisions
        # For ThunderSTORM, sigma-sigma, precision-uncertainty
        if self.parameters['method'] == 'GDSC SMLM 1':
            df = df.loc[df['X SD'] <= paras['sigma']]
            df = df.loc[df['Precisions (nm)'] <= paras['precision']]
        elif self.parameters['method'] == 'ThunderSTORM':
            df = df.loc[df['sigma [nm]'] <= paras['sigma']*self.parameters['pixel_size']]
            df = df.loc[df['uncertainty_xy [nm]'] <= paras['precision']]

        df = df.reset_index(drop=True)
        df.to_csv(file_dir.replace('.csv', '_filter_'+str(paras['precision'])+'_'+str(paras['sigma'])+'.csv')) # File naming: filter_precision_sigma

        return 1


    def _cluster_DBSCAN(self, field_name):
        # The result files are named differently for drift corrected and uncorrected data
        if self.path_result_fid.endswith('raw'):
            file_dir = os.path.join(self.path_result_fid, field_name+'_results.csv')
        else:
            file_dir = os.path.join(self.path_result_fid, field_name+'_corrected.csv')

        # Using  filtered data for clustering
        try:
            filters = self.parameters['filter']
            file_dir = file_dir.replace('.csv', '_filter_'+str(paras['precision'])+'_'+str(paras['sigma'])+'.csv')
        except KeyError:
            pass

        df = pd.read_csv(file_dir)
        # The coordinates for localisations are in pixel for GDSC results and nm for TS results. This step unifies the unit to nm.
        if self.parameters['method'] == 'GDSC SMLM 1':
            df['x [nm]'] = df['X'] * self.parameters['pixel_size']
            df['y [nm]'] = df['Y'] * self.parameters['pixel_size']
        elif self.parameters['method'] == "ThunderSTORM":
            df['X'] = df['x [nm]'] / self.parameters['pixel_size']
            df['Y'] = df['y [nm]'] / self.parameters['pixel_size']

        clustering = DBSCAN(eps=self.parameters['DBSCAN']['eps'] , min_samples=self.parameters['DBSCAN']['min_sample']).fit(df[['x [nm]', 'y [nm]']])

        labels = list(clustering.labels_)
        n_clusters = len(labels) - (1 if -1 in labels else 0)
        n_noise = labels.count(-1)

        # Label the localisations in the df with cluster ids
        labelled_df = df.copy()
        labelled_df['DBSCAN_label'] = labels

        # Remove localisations labelled as noise from the df
        cleaned_df = labelled_df.copy()
        cleaned_df = labelled_df[labelled_df.DBSCAN_label != -1]
        # Save cleaned localisation file
        cleaned_df.to_csv(os.path.join(self.path_result_fid, field_name+'_clustered.csv'))

        cleaned_labels = labels.copy()
        cleaned_labels = cleaned_labels[cleaned_labels != -1]

        # Magnify the coordinates
        df['X_mag'] = (df['X'] * self.parameters['scale']).astype(np.uint16)
        df['Y_mag'] = (df['Y'] * self.parameters['scale']).astype(np.uint16)

        # Cluster profiling
        placeholder = pd.DataFrame({
            'X_mag': np.tile(range(0, self.img_shape[1] * self.parameters['scale']), self.img_shape[2] * self.parameters['scale']), 
            # Repeat x coordinates y times
            'Y_mag': np.repeat(range(0, self.img_shape[2] * self.parameters['scale']), self.img_shape[1] * self.parameters['scale'])
            # Repeat y coordinates x times
        }) # Creates a dataframe contains all the pixel localisations
        
        cluster_df = cleaned_df.copy()
        cluster_df['DBSCAN_label'] += 1 # Change label 0 to 1
        cluster_df = pd.concat([cluster_df, placeholder], axis=0, join='outer') # Combine placeholder with actual dataframe
        cluster_df = pd.pivot_table(cluster_df, values='DBSCAN_label', index=['Y_mag'], columns=['X_mag'], aggfunc='max', fill_value=0) # Convert coordinate dataframe to array-like dataframe with index as Y, column as X, value as cluster label
        cluster_img = cluster_df.to_numpy()
        cluster_profile = regionprops_table(cluster_img, properties=['label', 'area', 'centroid', 'convex_area', 'major_axis_length', 'minor_axis_length', 'eccentricity','bbox'])
        cluster_profile = pd.DataFrame(cluster_profile)
        cluster_profile.columns(['cluster_id', 'no_localisation', 'X_(px)', 'Y_(px)', 'convex_area', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'xMin', 'yMin', 'xMax', 'yMax'])
        cluster_profile.to_csv(os.path.join(self.path_result_fid, field_name+'_clusterProfile.csv'))
        return 1


    def superRes_clustering(self, progress_signal=None):

        if progress_signal == None: #i.e. running in non-GUI mode
            workload = tqdm(sorted(self.fov_paths)) # using tqdm as progress bar in cmd
        else:
            workload = sorted(self.fov_paths)
            c = 0 # progress indicator

        for field in workload:
            
            try:
                self._cluster_dataFiltering(field)
            except KeyError:
                print('Filtering is not selected.')

            self._cluster_DBSCAN(field)

            if progress_signal == None:
                pass
            else:
                c += 1
                progress_signal.emit(c)

        return 1



if __name__ == "__main__":

    #path = input('Please input the path for analysis:\n')
    #if os.path.isdir(path) != True:
    #	print('Please input valid directory for data.')
    #	quit()
    project = SuperResAnalysis(r"D:\Work\Supres_test\Sample", {'method': 'GDSC SMLM 1'})
    #print('Launching: ' + path)
    #size = input('Please input the estimated size of particles(in pixels):\n')
    #threshold = input('Please input the threshold to apply(in nSD):\n')
    #print('Picking up particles in Fiji...')
    project.call_GDSC_SMLM()
    #print('Generating reports...')
    #project.generate_reports()
    #print('Done.')



