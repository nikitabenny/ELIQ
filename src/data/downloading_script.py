import wget
import os
import zipfile

def download_data (data_list):
    # write data_list links to a textfile
    with open('data_links.txt', 'w') as txt:
        for data in data_list:
            txt.write(f'{data}\n')
       

data_list = ['https://zenodo.org/record/7523691/files/M4RawV1.1_motion.zip?download=1',
             'https://zenodo.org/record/7523691/files/M4RawV1.1_multicoil_train.zip?download=1',
             'https://zenodo.org/record/7523691/files/M4RawV1.1_multicoil_val.zip?download=1'
             ]


# download_data(data_list)

# used 'wget -i C:\Users\1021624\ELIQ-Nikita\src\data\data_links.txt -P C:\Users\1021624\ELIQ-Nikita\data\raw' to download data
    #list of zip folders

def unzipper():
    folder_from = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw"
    folder_to = "C:\\Users\\1021624\\ELIQ-Nikita\\data\\raw_unzipped"

    items = os.listdir(folder_from) 
    abs_paths = [os.path.abspath(os.path.join(folder_from, item)) for item in items]
    print(abs_paths)

    #iterates through zip folders and opens each one
    for abs_path in abs_paths:
        with zipfile.ZipFile(abs_path,"r") as zip_ref:
            zip_ref.extractall(folder_to)