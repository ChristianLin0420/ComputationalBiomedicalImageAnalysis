import os
import numpy as np
from pydicom import dcmread
from matplotlib import pyplot as plt

### get DICOM file from folder
CT_chest_scans_path = "/Users/christianlin/Desktop/NTHU/Junior_2/Computational Methods for Biomedical Image Analysis/Homework/ComputationalBiomedicalImageAnalysis/Homework 1/CT_chest_scans"
CT_chest_folder_names = [name for name in os.listdir(CT_chest_scans_path)]
Fisrt_CT_Dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[0])]

# read DICOM file
ds = dcmread(CT_chest_scans_path + "/" + CT_chest_folder_names[0] + "/" + Fisrt_CT_Dataset[0])

slices = []
for i in range(len(Fisrt_CT_Dataset)):
    ds = dcmread(CT_chest_scans_path + "/" + CT_chest_folder_names[0] + "/" + Fisrt_CT_Dataset[i])
    slices.append(ds)

# list all metadata
print(ds)
plt.imshow(ds.pixel_array, interpolation='nearest')
plt.show()

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

first_patient_pixels = get_pixels_hu(slices)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()

image = ds.pixel_array.flatten()
rescaled_image = image * ds.RescaleSlope + ds.RescaleIntercept

print("Raw Data max value:                  {}".format(max(ds.pixel_array.flatten())))
print("Raw Data min value:                  {}".format(min(ds.pixel_array.flatten())))
print("Raw Data mean value:                 {}".format(np.mean(ds.pixel_array.flatten())))
print("Raw Data standard deviation value:   {}".format(np.std(ds.pixel_array.flatten())))

print("Raw Data max value:                  {}".format(max(rescaled_image.flatten())))
print("Raw Data min value:                  {}".format(min(rescaled_image.flatten())))
print("Raw Data mean value:                 {}".format(np.mean(rescaled_image.flatten())))
print("Raw Data standard deviation value:   {}".format(np.std(rescaled_image.flatten())))