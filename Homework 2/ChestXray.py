import os
import numpy as np

import csv
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import scikitplot as skplt
from alive_progress import alive_bar

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import sklearn.metrics as metrics

CT_chest_scans_path = "/Users/christianlin/Desktop/NTHU/Junior_2/Computational Methods for Biomedical Image Analysis/Homework/ComputationalBiomedicalImageAnalysis/Homework 2/chest_xray"
CT_chest_folder_names = [name for name in os.listdir(CT_chest_scans_path)]

test_dataset_filenames = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[0])]
train_normal_dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[1] + "/NORMAL")]
train_pneumonia_dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[1] + "/PNEUMONIA")]
validation_normal_dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[2] + "/NORMAL")]
validation_pneumonia_dataset = [f for f in os.listdir(CT_chest_scans_path + "/" + CT_chest_folder_names[2] + "/PNEUMONIA")]

def append_file_header(filenames, header):
    files = list()

    for filename in filenames:
        name = header + "/" + filename
        files.append(name)

    return files

def read_images(filenames):
    images = list()

    for filename in filenames:
        image = Image.open(filename)
        image = ImageOps.grayscale(image)
        image = np.asarray(image)
        if image.ndim == 3:
            print("dimension invalid")
            image = image.reshape(image.shape[0], -1)
        images.append(image)

    return images

test_dataset = append_file_header(test_dataset_filenames, CT_chest_scans_path + "/" + CT_chest_folder_names[0])
train_normal_dataset = append_file_header(train_normal_dataset, CT_chest_scans_path + "/" + CT_chest_folder_names[1] + "/NORMAL")
train_pneumonia_dataset = append_file_header(train_pneumonia_dataset, CT_chest_scans_path + "/" + CT_chest_folder_names[1] + "/PNEUMONIA")
validation_normal_dataset = append_file_header(validation_normal_dataset, CT_chest_scans_path + "/" + CT_chest_folder_names[2] + "/NORMAL")
validation_pneumonia_dataset = append_file_header(validation_pneumonia_dataset, CT_chest_scans_path + "/" + CT_chest_folder_names[2] + "/PNEUMONIA")

test_dataset = read_images(test_dataset)
train_normal_dataset = read_images(train_normal_dataset)
train_pneumonia_dataset = read_images(train_pneumonia_dataset)
validation_normal_dataset = read_images(validation_normal_dataset)
validation_pneumonia_dataset = read_images(validation_pneumonia_dataset)

print("test set amount: {}".format(len(test_dataset)))
print("train normal set amount: {}".format(len(train_normal_dataset)))
print("train pneumonia set amount: {}".format(len(train_pneumonia_dataset)))
print("validation normal set amount: {}".format(len(validation_normal_dataset)))
print("validation pneumonia set amount: {}".format(len(validation_pneumonia_dataset)))

'''
    Part 1
'''

# Original image metadata
sample_image = train_normal_dataset[0]
print("sample_image size: ", sample_image.shape)
print("sample_image min: ", min(sample_image.flatten()))
print("sample_image max: ", max(sample_image.flatten()))
print("sample_image mean: ", np.mean(sample_image.flatten()))

def resize_image(image):
    width , height = 512, 512
    resize_image = np.zeros(shape = (width, height))

    for W in range(width):
        for H in range(height):
            new_width = int( W * image.shape[0] / width )
            new_height = int( H * image.shape[1] / height )
            resize_image[W][H] = image[new_width][new_height]

    return resize_image / 255.0

processed_image = resize_image(sample_image)
        
print("processed_image size: ", processed_image.shape)
print("processed_image min: ", min(processed_image.flatten()))
print("processed_image max: ", max(processed_image.flatten()))
print("processed_image mean: ", np.mean(processed_image.flatten()))

'''
    Part 2
'''

# wirte csv file
def write_csv(patient_id, predictions):
    
    with open('107062240_HW2.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['case', 'predict'])

        for identify, result in zip(patient_id, predictions):
            writer.writerow([identify, result])

train_normal_dataset_label = np.zeros(len(train_normal_dataset))
train_pneumonia_dataset_label = np.ones(len(train_pneumonia_dataset))
validation_normal_dataset_label = np.zeros(len(validation_normal_dataset))
validation_pneumonia_dataset_label = np.ones(len(validation_pneumonia_dataset))

train_set = train_normal_dataset
validation_set = validation_normal_dataset

for data in train_pneumonia_dataset:
    train_set.append(data)

for data in validation_pneumonia_dataset:
    validation_set.append(data)

### Reshape all dataset images to size (512, 512)

print("start train set resize:")
with alive_bar(len(train_set)) as bar:
    for index in range(len(train_set)):
        train_set[index] = resize_image(train_set[index])
        bar()


print("start test set resize:")
with alive_bar(len(test_dataset)) as bar:
    for index in range(len(test_dataset)):
        test_dataset[index] = resize_image(test_dataset[index])
        bar()

print("start validation set resize:")
with alive_bar(len(validation_set)) as bar:
    for index in range(len(validation_set)):
        validation_set[index] = resize_image(validation_set[index])
        bar()

train_set_label = np.concatenate((train_normal_dataset_label, train_pneumonia_dataset_label))
validation_set_label = np.concatenate((validation_normal_dataset_label, validation_pneumonia_dataset_label))


### initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators = 200, random_state = 10, verbose = 1)


if len(train_set) == len(train_set_label):

    ### Shuffle the data set and the label set so that it can improve the training quality
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    train_set = np.array(train_set)
    test_dataset = np.array(test_dataset)
    validation_set = np.array(validation_set)

    train_set, train_set_label = unison_shuffled_copies(train_set, train_set_label)
    validation_set, validation_set_label = unison_shuffled_copies(validation_set, validation_set_label)

    print(train_set.shape)
    print(test_dataset.shape)
    print(validation_set.shape) 

    ntsamples, ntx, nty = train_set.shape
    nvsampels, nvx, nvy = validation_set.shape
    nesamples, nex, ney = test_dataset.shape

    ### Reshape the dataset as 1-dimensional data
    train_set = train_set.reshape(ntsamples, ntx * nty)
    test_dataset = test_dataset.reshape(nesamples, nex * ney)
    validation_set = validation_set.reshape(nvsampels, nvx * nvy)

    model.fit(train_set, train_set_label)
    
    ### Save model
    joblib.dump(model, 'RFC_model')

    validation_set_predict = model.predict(validation_set)
    validation_score = model.predict_proba(validation_set)
    validation_score = np.argmax(validation_score, axis=1)

    # Accuracy
    accuracy_score = accuracy_score(validation_set_label, validation_set_predict)
    # Sensitivity(Recall)
    recall_score = recall_score(validation_set_label, validation_set_predict, average='weighted')
    # Precision
    precision_score = precision_score(validation_set_label, validation_set_predict, average='weighted')
    # ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(validation_set_label, validation_score, pos_label = 2)

    # skplt.metrics.plot_roc(validation_set_label, validation_score)
    # plt.show()

    plt.savefig("ROC curve")

    print("Accuracy: {}".format(accuracy_score))
    print("Recall: {}".format(recall_score))
    print("Precision: {}".format(precision_score))
    print("fpr: {}".format(fpr))
    print("tpr: {}".format(tpr))
    print("thresholds: {}".format(thresholds))

    '''
        Part 3
    '''

    # Write csv
    test_set_predict = model.predict(test_dataset)
    write_csv(test_dataset_filenames, test_set_predict)
else:
    print("on shit")

