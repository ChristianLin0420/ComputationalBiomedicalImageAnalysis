import os
import numpy as np
import skimage
from pydicom import dcmread
from matplotlib import pyplot as plt

### Ploting ###
def plot_segmentation(original_image, seg_image):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1, adjustable='box')
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box')

    ax[0].imshow(original_image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].hist(original_image.ravel(), bins=256)
    ax[1].set_title('Histogram')

    ax[2].imshow(seg_image, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')

    plt.show()

### get DICOM file from folder
CT_chest_scans_path = "/Users/christianlin/Desktop/NTHU/Junior_2/Computational Methods for Biomedical Image Analysis/Homework/ComputationalBiomedicalImageAnalysis/Homework 1/CT_chest_scans"
CT_chest_folder_names = [name for name in os.listdir(CT_chest_scans_path)]
Fisrt_CT_Dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[0])]

# read DICOM file
ds = dcmread(CT_chest_scans_path + "/" + CT_chest_folder_names[0] + "/" + Fisrt_CT_Dataset[50])

# threshold method
threshold_mean = np.mean(ds.pixel_array.flatten())
threshold_median = np.median(ds.pixel_array.flatten())

print("Raw Data mean value:     {}".format(threshold_mean))
print("Raw Data median value:   {}".format(threshold_median))

#### Apply Segmentation ####

# Mean
original_image = ds.pixel_array
mean_image = np.zeros(shape=(512, 512))

for i in range(512):
    for j in range(512):
        if original_image[i][j] >= threshold_mean:
            mean_image[i][j] = 1.0
        else:
            mean_image[i][j] = 0.0

# Median
median_image = np.zeros(shape=(512, 512))

for i in range(512):
    for j in range(512):
        if original_image[i][j] >= threshold_median:
            median_image[i][j] = 1.0
        else:
            median_image[i][j] = 0.0

plot_segmentation(original_image, mean_image)
plot_segmentation(original_image, median_image)