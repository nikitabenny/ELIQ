import h5py
import numpy as np
from matplotlib import pyplot as plt
import os


# - h5 keys: 
#     - ismrmd_header --> utf-8 string of path?
#     - kspace data
#     - reconstruction_rss (root sum of squares transform and shows how to convert to get full images from multiple coils)


def show_keys(training_folder):
    items = os.listdir(training_folder)
    abs_paths = [os.path.abspath(os.path.join(training_folder, item)) for item in items]

    unique_keys = []

    for abs_path in abs_paths:
        hf = h5py.File(abs_path)
        keys =  list(hf.keys())
        for key in keys:
            if key not in unique_keys:
                unique_keys.append(key)
        
    print(unique_keys)



    

# training_folder = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped\\multicoil_train"




# volume_kspace = hf['kspace'][()]
# print('Kspace shape:', volume_kspace.shape)
# out = hf['reconstruction_rss'][()]
# print('reconstruction rss shape:',out.shape) #(number of slices, height, width)