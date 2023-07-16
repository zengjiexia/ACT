import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import tifffile as tiff

class SR_image:
    def __init__(self, SR_image_size, magnification, SR_pixel_size):
        # Image parameters
        self.SR_image_size = SR_image_size
        self.magnification = magnification
        self.SR_pixel_size = SR_pixel_size
        self.DL_image_size = self.SR_image_size/self.magnification
        self.DL_pixel_size = self.SR_pixel_size*self.magnification
            
        self.FWHM_DL_nm = 400
        self.FWHM_SR_pixel = int(self.magnification*self.FWHM_DL_nm/self.DL_pixel_size)

        # parameters for particles
        self.particle_shape = 'straight'
        self.number_of_particles = 50
        self.particle_size = 'auto'
        self.intensity_mean = 4000
        self.intensity_sigma = 0.2

        # parameters for noise
        self.noise_mean = 1000
        self.noise_sigma = 0.5

        # parameters for SR simulation generation
        self.number_of_frame = 1000
        self.max_blink_per_image = 2*self.number_of_particles
        self.percentage_leftover = 0.2

        # parameters for drift
        self.apply_drift = 1
        self.max_drift_per_frame = 5 # nm


    def set_particle_details(self, shape='straight', number=50, size='auto'):
        self.particle_shape = shape
        self.number_of_particles = number
        self.particle_size = size


    def set_intensity(self, mean, sigma):
        self.intensity_mean = mean
        self.intensity_sigma = sigma


    def set_noise(self, mean, sigma):
        self.noise_mean = mean
        self.noise_sigma = sigma


    def set_frame(self, number_of_frame):
        self.number_of_frame = number_of_frame


    def set_path_name(self, path, name):
        self.ground_truth_path = path + r'\ground_truth_images'
        if not os.path.isdir(self.ground_truth_path):
            os.mkdir(self.ground_truth_path)

        self.simulated_image_path = path + r'\simulated_SR_data'
        if not os.path.isdir(self.simulated_image_path):
            os.mkdir(self.simulated_image_path)
        
        self.filename = name


    def set_drift(self, apply_drift=1, max_drift_per_frame=5):
        self.apply_drift=apply_drift
        self.max_drift_per_frame=max_drift_per_frame


    def gaussian_dist(self, x, muu, sigma, amp=0):
        if amp == 0:
            amp = (1/(sigma*((2*np.pi)**(1/2))))
        return amp*np.exp(-0.5*((x-muu)/sigma)**2)


    def generate_particle_list(self):
        # Generate particle list formed by image of particle (in array format)
        # Generate position list for these particles
        def error_return():
            try:
                return self.particle_list
            except NameError:
                print('No previous list found')
                return 0
            
        if (self.particle_shape != 'straight') & (self.particle_shape != 'dot'): # prevent unknown shape
            print('Unknown shape, try to return the previous list')
            error_return()
            
        rotate_list = np.random.randint(0, 360, size = self.number_of_particles)
        
        # prepare self.size_nm_list
        if self.particle_size == 'auto':
            if self.particle_shape == 'straight':
                number_of_groups = np.random.randint(1, 6)
                particle_per_group = int(self.number_of_particles/number_of_groups)
                group_mean_size = np.random.randint(self.SR_image_size*self.SR_pixel_size*0.01,self.SR_image_size*self.SR_pixel_size*0.1, size=number_of_groups)
                self.size_nm_list = np.array([]) # size in nm
                for g in range(0, number_of_groups):
                    self.size_nm_list = np.concatenate((self.size_nm_list, np.random.normal(group_mean_size[g], group_mean_size[g]*0.10, size=particle_per_group)))
            elif self.particle_shape == 'dot':
                number_of_groups = np.random.randint(1, 6)
                particle_per_group = int(self.number_of_particles/number_of_groups)
                group_mean_size = np.random.randint(50, 200, size=number_of_groups)
                self.size_nm_list = np.array([]) # size in nm
                for g in range(0, number_of_groups):
                    self.size_nm_list = np.concatenate((self.size_nm_list, np.random.normal(group_mean_size[g], group_mean_size[g]*0.10, size=particle_per_group)))
        else: # fixed size
            try:
                self.size_nm_list = np.random.normal(self.particle_size, self.particle_size*0.10, size=self.number_of_particles)
            except TypeError:
                print('Unknown size, try to return the previous list')
                error_return()
        
        size_pixel_list = self.size_nm_list/self.SR_pixel_size
        self.size_pixel_sr_list = np.round(np.sqrt(0.5*(size_pixel_list**2))) # calculate the image for line by trigonometry
        self.size_pixel_sr_list = self.size_pixel_sr_list.astype(np.int32) # Then round and convert to int32
        self.number_of_particles = len(self.size_pixel_sr_list) # set number of particles to actual number of particles generated

        # prepare particle_list
        self.particle_list = []
        if self.particle_shape == 'straight':
            for i in range(0, self.number_of_particles):
                side = self.size_pixel_sr_list[i]
                shape = [(0, 0), (side, side)]
                canvas = Image.new("I", (side, side))
                img = ImageDraw.Draw(canvas)  
                img.line(shape, width = 2)
                canvas = canvas.rotate(angle=rotate_list[i], expand=True, resample=Image.Resampling.NEAREST)
                img_arr = np.asarray(canvas)
                nz = np.nonzero(img_arr)
                img_arr = img_arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
                self.particle_list.append(img_arr)
        elif self.particle_shape == 'dot':
            for i in range(0, self.number_of_particles):
                side = self.size_pixel_sr_list[i]
                shape = [(0, 0), (side-1, side-1)]
                canvas = Image.new("I", (side, side))
                img = ImageDraw.Draw(canvas) 
                img.ellipse(shape, fill=1)
                img_arr = np.asarray(canvas)
                self.particle_list.append(img_arr)
        return self.particle_list


    def generate_particle_positions(self):
        # generate position list for particles (left top as 0,0)
        self.x_list = np.random.randint(0, self.SR_image_size, size=self.number_of_particles)
        self.y_list = np.random.randint(0, self.SR_image_size, size=self.number_of_particles)
        return self.x_list, self.y_list


    def draw_groundtruth(self):
        # save the ground truth data for reference
        simulation_details = pd.DataFrame({
            'x_pixel': self.x_list,
            'y_pixel': self.y_list,
            'length': self.size_nm_list,
            'sr_pixel_size': self.size_pixel_sr_list
            })
        simulation_details.to_csv(self.ground_truth_path + '\\' + self.filename + '.csv')

        # Create a empty SR image canvas with set size
        # Plot all the particle generated onto the canvas
        self.SR_groundtruth_image = np.zeros([self.SR_image_size, self.SR_image_size], dtype = np.int32)
        canvas_shape = self.SR_groundtruth_image.shape
        for i in range(0, self.number_of_particles):
            x, y = self.x_list[i], self.y_list[i]
            particle = self.particle_list[i]
            particle_shape = particle.shape
            if x + particle_shape[0] >= canvas_shape[0]:
                x -= particle_shape[0]
            if y + particle_shape[1] >= canvas_shape[1]:
                y -= particle_shape[1]
            self.SR_groundtruth_image[x:x+particle_shape[0], y:y+particle_shape[1]] += particle
        img = Image.fromarray(self.SR_groundtruth_image)
        img.save(self.ground_truth_path + '\\' + self.filename + '.tif')
        return self.SR_groundtruth_image


    def generate_blinks(self):
        # Generate blink list
            # Find the non-zero indices and values in the array
        nonzero_indices = np.nonzero(self.SR_groundtruth_image)
        values = self.SR_groundtruth_image[nonzero_indices]
            # Create a list of coordinates based on the values
        self.blink_list = np.array([coord for coord, value in zip(zip(*nonzero_indices), values) for _ in range(value)])
        
        return self.blink_list


    def generate_base_image(self):
        self.base_simulated_image = np.zeros([self.SR_image_size, self.SR_image_size], dtype = np.int32)
        return self.base_simulated_image


    def add_fiducials(self, number_of_fiducial=3, fiducial_brightness_rate=5):
        """
        paras:
        number_of_fiducial: int;
        fiducial_brightness_rate: float, in terms of x times the intensity of signal.
        """
        fid_x_pos = np.random.randint(int(self.SR_image_size*0.1), int(self.SR_image_size*0.9), size=number_of_fiducial)
        fid_y_pos = np.random.randint(int(self.SR_image_size*0.1), int(self.SR_image_size*0.9), size=number_of_fiducial)
        
        # define fiducial
        x, y = np.meshgrid(np.linspace(-1,1,self.FWHM_SR_pixel*2), np.linspace(-1,1,self.FWHM_SR_pixel*2))
        dst = np.sqrt(x*x+y*y)
        gauss_muu = 0
        gauss_sigma = 0.3
        fiducial = self.gaussian_dist(dst, gauss_muu, gauss_sigma, amp=1)
        fiducial *= self.intensity_mean*fiducial_brightness_rate
        fiducial = fiducial.astype('int32')
        
        # add fiducial
        for i in range(0, number_of_fiducial):
            self.base_simulated_image[int(fid_x_pos[i]-self.FWHM_SR_pixel):int(fid_x_pos[i]+self.FWHM_SR_pixel), int(fid_y_pos[i]-self.FWHM_SR_pixel):int(fid_y_pos[i]+self.FWHM_SR_pixel)] += fiducial
        return self.base_simulated_image


    def draw_blinks(self, leftover=[]):
        blinked_image = self.base_simulated_image.copy()
        number_of_blinks = np.random.randint(1, 3*self.number_of_particles) # randomly pick the number of blink in a frame
        blink_positions = self.blink_list[np.random.choice(np.shape(self.blink_list)[0], size=number_of_blinks, replace=False)]# get blinks from blink list
        if len(leftover) != 0: # addtion of leftover blinks from previous frame
            blink_positions = np.concatenate((blink_positions, leftover), axis=0)
            number_of_blinks = len(blink_positions)

        intensities = np.random.normal(self.intensity_mean, self.intensity_mean*self.intensity_sigma, size=number_of_blinks)
        sizes = np.random.normal(1.1*self.FWHM_SR_pixel, 0.1*self.FWHM_SR_pixel, size=number_of_blinks).astype('int32')

        for i in range(0, number_of_blinks):
            x, y = np.meshgrid(np.linspace(-1,1,sizes[i]*2), np.linspace(-1,1,sizes[i]*2))
            dst = np.sqrt(x*x+y*y)
            gauss_muu = 0
            gauss_sigma = 0.3
            particle = self.gaussian_dist(dst, gauss_muu, gauss_sigma, amp=1) * intensities[i] # light up the particle with gaussian blur
            particle = particle.astype('int32')
            
            # plot particle onto base image
            pos_x, pos_y = blink_positions[i]
            if pos_x-sizes[i] < 0 or pos_y-sizes[i]<0 or pos_x+sizes[i]>self.SR_image_size or pos_y+sizes[i]>self.SR_image_size:
                pass
            else:
                blinked_image[pos_x-sizes[i]:pos_x+sizes[i], pos_y-sizes[i]:pos_y+sizes[i]] += particle
        if self.percentage_leftover != 0:
            leftover = blink_positions[np.random.choice(np.shape(blink_positions)[0], size=int(self.percentage_leftover*np.shape(blink_positions)[0]), replace=False)]
        return blinked_image, leftover


    def generate_drift(self):
        # define cummulative drift
        x_trajectory_nm = np.cumsum(np.random.choice([-self.max_drift_per_frame, self.max_drift_per_frame], size=self.number_of_frame))
        y_trajectory_nm = np.cumsum(np.random.choice([-self.max_drift_per_frame, self.max_drift_per_frame], size=self.number_of_frame))
        self.x_drift_pixel = np.round(x_trajectory_nm/self.SR_pixel_size).astype(np.int32)
        self.y_drift_pixel = np.round(y_trajectory_nm/self.SR_pixel_size).astype(np.int32)
        return self.x_drift_pixel, self.y_drift_pixel


    def draw_drift(self, blinked_image, frame_count):
        x_drift = self.x_drift_pixel[frame_count]
        y_drift = self.y_drift_pixel[frame_count]

        x_padding = np.zeros((abs(x_drift), self.SR_image_size))
        y_padding = np.zeros((self.SR_image_size, abs(y_drift)))

        drifted_image = blinked_image.copy()
        if x_drift > 0:
            drifted_image = np.concatenate((x_padding, drifted_image), axis=0)
            drifted_image = drifted_image[:self.SR_image_size, :]
        elif x_drift < 0:
            drifted_image = np.concatenate((drifted_image, x_padding), axis=0)
            drifted_image = drifted_image[-x_drift:, :]
        else:
            pass

        if y_drift > 0:
            drifted_image = np.concatenate((y_padding, drifted_image), axis=1)
            drifted_image = drifted_image[:, :self.SR_image_size]
        elif y_drift < 0:
            drifted_image = np.concatenate((drifted_image, y_padding), axis=1)
            drifted_image = drifted_image[:, -y_drift:]
        else:
            pass

        return drifted_image

    def create_SR_stack(self):
        SR_stack = []
        if self.apply_drift == 1:
            self.generate_drift()
        leftover = []
        for f in range(0, self.number_of_frame):
            frame, leftover = self.draw_blinks(leftover=leftover)
            if self.apply_drift == 1:
                frame = self.draw_drift(blinked_image=frame, frame_count=f)
            # reshape SR image to DL size
            DL_frame = frame.reshape(int(self.DL_image_size), self.magnification, int(self.DL_image_size), self.magnification).mean(-1).mean(1)
            noise = np.random.normal(self.noise_mean, self.noise_mean*self.noise_sigma, size=np.shape(DL_frame))
            DL_frame = DL_frame + noise
            SR_stack.append(DL_frame)
        tiff.imwrite(self.simulated_image_path+'\\'+self.filename+'.tif', SR_stack)
        return SR_stack