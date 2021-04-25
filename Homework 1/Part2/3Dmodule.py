import os
import numpy as np
from pydicom import dcmread
from matplotlib import pyplot as plt

##### Normalization #####
def normalize(sl):
    image = sl.pixel_array
    image += 1024
    image = image / 4096.0
    return image

##### Functions #####
def plot_slices(slices):
    fig = plt.figure(figsize = (10, 10))
    columns = 5
    rows = 5
    
    for i in range(1, columns * rows + 1):
        img = slices_pixel_arrays[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap = "gray")
    
    plt.show()

### get DICOM file from folder
FOLDER_INDEX = 0
CT_chest_scans_path = "/Users/christianlin/Desktop/NTHU/Junior_2/Computational Methods for Biomedical Image Analysis/Homework/ComputationalBiomedicalImageAnalysis/Homework 1/CT_chest_scans"
CT_chest_folder_names = [name for name in os.listdir(CT_chest_scans_path)]
Fisrt_CT_Dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[FOLDER_INDEX])]

slices = []
for i in range(len(Fisrt_CT_Dataset)):
    ds = dcmread(CT_chest_scans_path + "/" + CT_chest_folder_names[FOLDER_INDEX] + "/" + Fisrt_CT_Dataset[i])
    slices.append(ds)

slices = sorted(slices, key=lambda s: s.SliceLocation)

# Visualize montage of slices.
slices_pixel_arrays = []
for i in range(25):
    slices_pixel_arrays.append(slices[i + 25].pixel_array)

# plot_slices(slices_pixel_arrays)
normalize_images = []
for i in range(25):
    normalize_images.append(normalize(slices[i + 50]))

plot_slices(normalize_images)