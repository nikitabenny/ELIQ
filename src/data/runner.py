import h5py
from data_acquisition import show_keys
import numpy as np

# training_folder = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped\\multicoil_train"

# show_keys(training_folder) #only keys are ismrmrd_header, kspace and reconstruction rss
hf = h5py.File("C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped\\multicoil_train\\2022111802_T101.h5") 
volume_kspace = hf['kspace'][()] #volume_kspace is a numpy array with shape (18,4,256,256)
shape_kspace = volume_kspace.shape



# shape of space is 4x2x2x3 
# real shape of kspace is 18x4x256x256  = # of images(slices)  x (# of coils) x (256x256 pixel size of images)

space = [
[
    [    
    [5,4,3],
    [2,4,6],
    ],
    [
    [5,4,3],
    [2,4,6],
    ]
],

[
    [    
    [5,4,3],
    [2,4,6],
    ],
    [
    [5,4,3],
    [2,4,6],
    ]
],

[
    [    
    [5,4,3],
    [2,4,6],
    ],
    [
    [5,4,3],
    [2,4,6],
    ]
],

[
    [    
    [5,4,3],
    [2,4,6],
    ],
    [
    [5,4,3],
    [2,4,6],
    ]
]


]

arr = np.array(space)
print(type(arr))
print(arr.shape)
print(arr)

