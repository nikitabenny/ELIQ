#need to use k means clustering to check centroids and view skew of data and distribution based on needed variables

#need to visualize data first 

import fastmri
import os
import h5py
import numpy as np
from fastmri.data import transforms as T
import matplotlib.pyplot as plt
import matplotlib
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFractionFunc



data_folder = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped\\multicoil_train"

items = os.listdir(data_folder) 
abs_paths = [os.path.abspath(os.path.join(data_folder, item)) for item in items]

# plots absolute value of kspace
def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)

    plt.show()


path = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped\\multicoil_train\\2022061003_T201.h5"
hf = h5py.File(path)
volume_kspace = hf['kspace'][()]
slice_kspace = volume_kspace[8] 
show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 1, 2, 3])  # This shows coils 0, 1, 2 and 3



slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image


show_coils(slice_image_abs, [0, 1, 2, 3], cmap='gray')


mask_func = EquispacedMaskFractionFunc(center_fractions=[0.04], accelerations=[3])  # Create the mask function object
masked_kspace, mask, _ = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
plt.show()




# def visualizer():
#     folder_from = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw"
#     folder_to = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\processed\\k-space"

#     items = os.listdir(folder_from) 
#     abs_paths = [os.path.abspath(os.path.join(folder_from, item)) for item in items]
    
#     for path in abs_paths:
#         for i in range(0,18):    
#             hf = h5py.File(path)
#             volume_kspace = hf['kspace'][()]
#             slice_kspace = volume_kspace[i] 
        

