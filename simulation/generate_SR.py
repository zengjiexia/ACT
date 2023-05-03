# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:34:18 2022

@author: Trevor Wu
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


## parameters for images
DL_img_size = 500
mag = 5
SR_img_size = DL_img_size * mag
DL_pixel_size = 100 # nm
FWHM = 400 # nm
FWHM_px_SR = int(mag*FWHM/DL_pixel_size)
mean_intensity = 4000
sigma_intensity = 0.2
noise_intensity = 1000
sigma_noise = 0.5
num_frames = 1000

## parameters for aggregates
num_straight = 50         # number of straight fibrils
num_dot = 50               # number of dots
num_curly = 50             # number of curly fibrils

total_num = num_straight + num_dot + num_curly

len_straight = 20        # pixels in DL img
dia_dot = 1              # pixels in DL img
len_curly = 100           # pixels in DL img

sigma = 0.1              # variation in size of all three types of aggregates

# drift
min_move = 5 #nm, drift in x or y between frames

# select blinking spots - some dots remains for more than one frame
num_blink = 100 # num of dots on one frame, 100
num_lasting = 20 # num of dots blinking till the next frame, must be smaller than num_blink, 20
num_fid = 2 # 3 fiducial markers in each frame
intensity_fid = 5*mean_intensity # fiducials are 5 times as bright as normal fluorophores

# folder for DL images (blinkings)
path = r'I:\Data\20220714_fake_images\test\im4\raw'
# name of ground truth super-res image
SR_name = path+r"\SR.tif"

SR_len_straight = len_straight * mag
SR_dia_dot = dia_dot * mag
SR_len_curly = len_curly * mag

straight_len_list1 = np.random.normal(SR_len_straight, sigma*SR_len_straight, num_straight)
straight_len_list2 = np.random.normal(SR_len_straight, sigma*SR_len_straight, num_straight)
dot_dia_list1 = np.random.normal(SR_dia_dot, sigma*SR_dia_dot, num_dot)
dot_dia_list2 = np.random.normal(SR_dia_dot, sigma*SR_dia_dot, num_dot)
curly_len_list1 = np.random.normal(SR_len_curly, sigma*SR_len_curly, num_curly)
curly_len_list2 = np.random.normal(SR_len_curly, sigma*SR_len_curly, num_curly)

straight_len_list1 = straight_len_list1.astype(np.int32)
straight_len_list2 = straight_len_list2.astype(np.int32)
dot_dia_list1 = dot_dia_list1.astype(np.int32)
dot_dia_list2 = dot_dia_list2.astype(np.int32)
curly_len_list1 = curly_len_list1.astype(np.int32)
curly_len_list2 = curly_len_list2.astype(np.int32)

x_list = np.random.randint(0, SR_img_size, size = total_num)
y_list = np.random.randint(0, SR_img_size, size = total_num)

rotate_list = np.random.randint(0, 360, size = num_straight)

# for curly fibrils, additional info about its shape is needed
min_angle = 10
max_angle = 30
start_angle_list = np.random.randint(0, 360, size = num_curly)
angle_list = np.random.randint(min_angle, max_angle, size = num_curly)
end_angle_list = start_angle_list + angle_list

# draw straight lines in an image
straight_list = []
for i in range(0, num_straight):
    w, h = straight_len_list1[i], straight_len_list2[i]
    shape = [(0, 0), (w, h)]
    img_straight = Image.new("I", (w, h))
    img1_straight = ImageDraw.Draw(img_straight)  
    img1_straight.line(shape, width = 2)
    img_straight = img_straight.rotate(rotate_list[i])
    img_straight_arr = np.asarray(img_straight)
    nz = np.nonzero(img_straight_arr)
    straight_arr = img_straight_arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
    straight_list.append(straight_arr)

# draw dots in an image
dot_list = []
for i in range(0, num_dot):
    w, h = dot_dia_list1[i], dot_dia_list2[i]
    shape = [(1, 1), (w-1, h-1)]
    img_dot = Image.new("I", (w, h))
    img1_dot = ImageDraw.Draw(img_dot)  
    img1_dot.ellipse(shape, fill = 1)
    dot_arr = np.asarray(img_dot)
    dot_list.append(dot_arr)

# draw curly fibril in an image
curly_list = []
for i in range(0, num_curly):
    w, h = curly_len_list1[i], curly_len_list2[i]
    shape = [(1, 1), (w, h)]
    img_curly = Image.new("I", (w-1, h-1))
    img1_curly = ImageDraw.Draw(img_curly)  
    img1_curly.arc(shape, start = start_angle_list[i], end = end_angle_list[i])
    img_curly_arr = np.asarray(img_curly)
    nz = np.nonzero(img_curly_arr)
    curly_arr = img_curly_arr[nz[0].min():nz[0].max()+1, nz[1].min():nz[1].max()+1]
    curly_arr = ndimage.binary_dilation(curly_arr)
    curly_list.append(curly_arr)

def crop_img(small_img, large_img, x, y):
    small_size = np.shape(small_img)
    large_size = np.shape(large_img)
    if x + small_size[0] >= large_size[0]:
        small_img = small_img[0:(large_size[0] - x -1), :]
    if y + small_size[1] >= large_size[1]:
        small_img = small_img[:, 0:large_size[1] - y -1]
    return small_img

SR_blank = np.zeros([SR_img_size, SR_img_size], dtype = np.int32)
for i in range(0, num_straight):
    x, y = x_list[i], y_list[i]
    straight_img = crop_img(straight_list[i], SR_blank, x, y)
    SR_blank[x:x+np.shape(straight_img)[0], y:y+np.shape(straight_img)[1]] += straight_img

for i in range(0, num_dot):
    x, y = x_list[num_straight+i], y_list[num_straight+i]
    dot_img = crop_img(dot_list[i], SR_blank, x, y)
    SR_blank[x:x+np.shape(dot_img)[0], y:y+np.shape(dot_img)[1]] += dot_img
    
for i in range(0, num_curly):
    x, y = x_list[num_straight+num_dot+i], y_list[num_straight+num_dot+i]
    curly_img = crop_img(curly_list[i], SR_blank, x, y)
    SR_blank[x:x+np.shape(curly_img)[0], y:y+np.shape(curly_img)[1]] += curly_img
    
SR_img = Image.fromarray(SR_blank)
SR_img.save(SR_name)

# get coordinates of binding sites
index_list = list(np.stack(np.nonzero(SR_img), axis = -1))

# get coordinates of multiple binding sites, add into the index list
SR_array = np.copy(SR_blank)
multiple_sites = np.stack(np.where(SR_array>1), axis = -1)
for i in range(0, np.shape(multiple_sites)[0]):
    num_sites = SR_array[tuple(multiple_sites[i])]
    for j in range(0, num_sites-1):
        index_list.append(multiple_sites[i])



# define fiducials 
x_fid = np.random.randint(int(SR_img_size*0.1), int(SR_img_size*0.9), size = num_fid)
y_fid = np.random.randint(int(SR_img_size*0.1), int(SR_img_size*0.9), size = num_fid)

# define drift trajectory

x_steps = np.random.choice([-min_move, min_move], size = num_frames) + 0.1 * np.random.randn(num_frames)
y_steps = np.random.choice([-min_move, min_move], size = num_frames) + 0.1 * np.random.randn(num_frames)
x_drift = np.cumsum(x_steps)
y_drift = np.cumsum(y_steps)
x_drift_SR = np.round(x_drift/DL_pixel_size*mag).astype(np.int32)
y_drift_SR = np.round(y_drift/DL_pixel_size*mag).astype(np.int32)

# for first frame
blink_index = list(np.random.randint(0, len(index_list), num_blink))
blink = []
blink.append(blink_index)

for i in range(1, num_frames):
    remain_index = np.random.randint(0, num_blink, num_lasting) #from 100 blinks on the previous frame, take out 50 indices
    new_index = np.random.randint(0, len(index_list), num_blink-num_lasting) # from the whole list, take out 100-50 indices
    blink_list = []
    for j in range(0, num_lasting):
        blink_list.append(blink_index[remain_index[j]])
    for k in range(0, num_blink-num_lasting):
        blink_list.append(new_index[k])
    blink.append(blink_list)
    blink_index = blink_list.copy()

def create_frame(blink, num_blink, frame, index_list, SR_img_size, FWHM_px_SR, path):
    blank = np.zeros((SR_img_size, SR_img_size))
    list_intensity = np.random.normal(mean_intensity, mean_intensity*sigma_intensity, size = num_blink)
    list_dia = np.random.normal(FWHM_px_SR*1.1, FWHM_px_SR*0.1, size = num_blink)
    list_dia = list_dia.astype(np.int16)
    for i in range(0, num_blink):
        x, y = np.meshgrid(np.linspace(-1,1,list_dia[i]*2), np.linspace(-1,1,list_dia[i]*2))
        dst = np.sqrt(x*x+y*y)
        # Initializing sigma and muu
        sigma = 0.3
        muu = 0
        # Calculating Gaussian array
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        gauss_intensity = gauss*list_intensity[i]
        dot_size = int(list_dia[i])
        coord_x, coord_y = index_list[int(blink[frame][i])][0], index_list[int(blink[frame][i])][1]
        if coord_x-dot_size < 0 or coord_y-dot_size<0 or coord_x+dot_size>SR_img_size or coord_y+dot_size>SR_img_size:
            pass
        else:
            blank[coord_x-dot_size:coord_x+dot_size, coord_y-dot_size:coord_y+dot_size] += gauss_intensity
    SR_frame = blank.copy()
    return SR_frame

def add_fiducial(SR_frame, num_fid, x_fid, y_fid, intensity_fid, FWHM_px_SR):
    SR_frame_addfid = SR_frame.copy()
    for i in range(0, num_fid):
        x, y = np.meshgrid(np.linspace(-1,1,FWHM_px_SR*2), np.linspace(-1,1,FWHM_px_SR*2))
        dst = np.sqrt(x*x+y*y)
        # Initializing sigma and muu
        sigma = 0.3
        muu = 0
        # Calculating Gaussian array
        gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        gauss_intensity = gauss*intensity_fid
        SR_frame_addfid[int(x_fid[i]-FWHM_px_SR) : int(x_fid[i]+FWHM_px_SR), int(y_fid[i]-FWHM_px_SR):int(y_fid[i]+FWHM_px_SR)] += gauss_intensity
    SR_frame_fid = SR_frame_addfid.copy()
    return SR_frame_fid

def add_drift(SR_frame_fid, x_drift_SR, y_drift_SR, frame):
    x_d = x_drift_SR[frame]
    y_d = y_drift_SR[frame]
    x_size = np.shape(SR_frame_fid)[0]
    y_size = np.shape(SR_frame_fid)[1]
    if x_d == 0:
        frame_x = SR_frame_fid.copy()
    elif x_d > 0:
        frame_x = SR_frame_fid.copy()
        padding_x = np.zeros((x_d, y_size))
        frame_drift_x = np.concatenate((padding_x, frame_x), axis = 0)
        frame_x = frame_drift_x[:x_size, :]
    elif x_d < 0:
        frame_x = SR_frame_fid.copy()
        padding_x = np.zeros((-x_d, y_size))
        frame_drift_x = np.concatenate((frame_x, padding_x), axis = 0)
        frame_x = frame_drift_x[-x_d:, :]
    if y_d == 0:
        frame_y = frame_x.copy()
    elif y_d > 0:
        frame_y = frame_x.copy()
        padding_y = np.zeros((x_size, y_d))
        frame_drift_y = np.concatenate((padding_y, frame_y), axis = 1)
        frame_y = frame_drift_y[:, :y_size]
    elif y_d < 0:
        frame_y = frame_x.copy()
        padding_y = np.zeros((x_size,-y_d))
        frame_drift_y = np.concatenate((frame_y, padding_y), axis = 1)
        frame_y = frame_drift_y[:, -y_d:]
    return frame_y

def add_noise(DL_frame, noise_intensity, sigma_noise):
    noise = np.random.normal(noise_intensity, sigma_noise*noise_intensity, size = np.shape(DL_frame))
    noisy_frame = DL_frame + noise
    return noisy_frame

for frame in range(0, num_frames):
    SR_frame = create_frame(blink, num_blink, frame, index_list, SR_img_size, FWHM_px_SR, path)
    SR_frame_fid = add_fiducial(SR_frame, num_fid, x_fid, y_fid, intensity_fid, FWHM_px_SR)
    frame_drift = add_drift(SR_frame_fid, x_drift_SR, y_drift_SR, frame)
    DL_frame = frame_drift.copy()
    DL_frame = DL_frame.reshape(int(DL_frame.shape[0]/mag), mag, int(DL_frame.shape[1]/mag), mag).mean(-1).mean(1)
    noisy_frame = add_noise(DL_frame, noise_intensity, sigma_noise)
    final_frame = Image.fromarray(noisy_frame)
    frame_name = path + '\\'+ str(frame) + '.tif'
    final_frame.save(frame_name)