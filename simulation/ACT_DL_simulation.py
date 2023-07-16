# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:16:52 2022

@author: wyzzh
"""

# Importing Numpy package
import numpy as np
from scipy import ndimage
from PIL import Image


img_size = (512, 512) # size of the image in px
num_dots = 2800 # number of dots in the FOV
max_radius = 10 # max radius of signal dot, in px. If fibril, >=20. If super-res dots, recommended 10
avg_intensity = 4000 # average intensity of signals
intensity_variation = 1000 # intensity of all dots will be normally distributed with u = avg_intensity and sigma = variation
random_eccentricity = False # if True, eccentricity will be random. If false, eccentricity will be below the given value
eccentricity = 1 # 0.2 for fibril, 1 for dots
sharp_dots = True # if enabled, eccentricity will be forced to set to 1 (round dots only). Recommended radius = 10
sigma_illumination = 0.8 # sigma of the illumination profile of the FOV, centered and gaussian. This is to simulate a gaussian illumination
base_noise = 6000 # illumination-independent noise level
base_noise_sigma = 2000 # sigma of the baseline noise
illumination_noise = 3000  # illumination-dependent noise level

path = r'I:\Data\20220607_simulation_images\DL\density\im18'




# Generate file names
ideal_img_name = path + '\\ideal.tif'
illumi_bias_name = path + '\\illuminated.tif'
base_noise_name = path + '\\base_noise.tif'
illumination_noise_name = path + '\\X0Y0R1W1C0.tif'

radius_list_1 = np.random.randint(5, max_radius, size = num_dots)
if random_eccentricity == True:
    radius_list_2 = np.random.randint(5, max_radius*eccentricity, size = num_dots)
else:
    radius_list_2 = radius_list_1*eccentricity
    radius_list_2 = radius_list_2.astype(np.int32)
if sharp_dots == True:
    min_radius = int(max_radius * 0.7) # if sharp dots, min radius is 0.7 * max radius
    radius_list_1 = np.random.randint(min_radius, max_radius, size = num_dots)
    radius_list_2 = np.copy(radius_list_1)
rotate_list = np.random.randint(1, 180, size = num_dots)
intensity_list = np.random.normal(avg_intensity, intensity_variation, size = num_dots)
intensity_list[intensity_list<0]=0
pos_x_list = np.random.randint(0, img_size[0], size = num_dots)
pos_y_list = np.random.randint(0, img_size[1], size = num_dots)
dot_list = []

for i in range(0, num_dots):
    
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1,1,radius_list_1[i]), np.linspace(-1,1,radius_list_2[i]))
    dst = np.sqrt(x*x+y*y)
    # Initializing sigma and muu
    sigma = 0.3
    muu = 0.000
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    gauss_intensity = gauss*intensity_list[i]
    if sharp_dots == True:
        gauss_intensity = gauss_intensity.copy() * 4
    # Generate rotated dot
    if sharp_dots == False:
        gauss_rotate = ndimage.rotate(gauss_intensity, rotate_list[i], reshape = True)
        dot_list.append(gauss_rotate)
    else:
        dot_list.append(gauss_intensity)

ideal_img = np.zeros(img_size)
for i in range(0, num_dots):
    pos_x = pos_x_list[i]
    pos_y = pos_y_list[i]
    dot_x = np.shape(dot_list[i])[0]
    dot_y = np.shape(dot_list[i])[1]
    dot = dot_list[i]
    blank_img = np.zeros(img_size)
    if pos_x + dot_x > img_size[0]:
        cut_x = pos_x + dot_x - img_size[0]
        dot = dot[0:-cut_x, :]
    if pos_y + dot_y > img_size[1]:
        cut_y = pos_y + dot_y - img_size[0]
        dot = dot[:, 0:-cut_y]
    blank_img[pos_x:pos_x+np.shape(dot)[0], pos_y:pos_y+np.shape(dot)[1]] = dot
    ideal_img = ideal_img + blank_img
    
ideal_img = ideal_img.astype(np.uint32)
im = Image.fromarray(ideal_img)
im.save(ideal_img_name)

#illumination bias
x_bg, y_bg = np.meshgrid(np.linspace(-1,1,img_size[0]), np.linspace(-1,1,img_size[1]))
dst_bg = np.sqrt(x_bg*x_bg+y_bg*y_bg)
# Calculating Gaussian array
gauss_ill_bg = np.exp(-( (dst_bg-muu)**2 / ( 2.0 * sigma_illumination**2 ) ) )

illumination_img = ideal_img*gauss_ill_bg
illumination_img = illumination_img.astype(np.uint32)
im2 = Image.fromarray(illumination_img)
#im2.save(illumi_bias_name)

#baseline noise
base_img = np.random.normal(base_noise, base_noise_sigma, size = img_size)
base_noise_img = illumination_img + base_img
base_noise_img = base_noise_img.astype(np.uint16)
base_noise_img[base_noise_img<0] = 0
im3 = Image.fromarray(base_noise_img)
#im3.save(base_noise_name)

#illumination noise
illumination_noise_img = gauss_ill_bg * illumination_noise
final_img = illumination_noise_img + base_noise_img
final_img = final_img.astype(np.uint16)
final_img[final_img>55000] = base_noise
im4 = Image.fromarray(final_img)
im4.save(illumination_noise_name)

