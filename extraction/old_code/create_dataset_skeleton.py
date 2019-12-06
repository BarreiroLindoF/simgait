import os
import random

import numpy as np

from analysis import (display_clinical_stats, get_aligned_data,
                      get_clinical_data)
from parse_markers import reduce_data

# markers_names_final = ["LTOE", "LHEE", "LANK", "LKNE", "LASI", "LWRA_LWRB",
#                        "LELB", "LSHO", "LASI_RASI", "CLAV_C7_LSHO_RSHO",
#                        "LFHD_LBHD_RFHD_RBHD", "RWRA_RWRB", "RTOE", "RHEE",
#                        "RANK", "RKNE", "RASI", "RELB", "RSHO"]

markers_names_final = ["LTOE", "LHEE", "LANK", "LKNE", "LASI", "LSHO", "LASI_RASI",
                       "CLAV_C7_LSHO_RSHO", "LFHD_LBHD_RFHD_RBHD", "RTOE", "RHEE", "RANK", "RKNE", "RASI", "RSHO"]

currPath = os.path.dirname(os.getcwd())
path_to_data_markers = currPath + "\data\cleaned\CP\markers_aligned"
path_to_folder = currPath + "\data\cleaned\CP"
path_to_save = currPath + "\data\cleaned\CP\dataset"
dataset_name = "armless"
folder = "CP_hemiplegia_cycle_{}_markers".format(dataset_name)

all_cord = True


# Get data, labels (0 or 1) and files
data, labels, files_data = get_aligned_data(
    path_to_data_markers, path_to_folder, "Right")
data = reduce_data(data, markers_names_final)
print("Markers:", data.shape, labels.shape, len(files_data))


# Get clinical data
path_to_clinical_nn = currPath + "\data\extracted\CP"
clinical, features_clinical, files_clinical = get_clinical_data(
    path_to_clinical_nn, rows_max_nan=10, cols_max_nan=25, verbose=False)
print("Clinical:", clinical.shape, files_clinical.shape)

# keep only files that are in clinical
idx = [True if file_ in files_clinical.tolist() else False for file_ in files_data]
data = data[idx]
labels = labels[idx]
files_data = np.array(files_data)[idx]
print("Markers after remove:", data.shape, labels.shape, len(files_data))

# build matrix with clinical data for each file
data_clinical = np.zeros((data.shape[0], clinical.shape[1]))
for idx in range(data.shape[0]):
    data_clinical[idx, :] = clinical[files_clinical.tolist().index(
        files_data[idx]), :]

# remove NaNs
print("Before NaN's markers {}, clinical {}".format(np.count_nonzero(
    np.isnan(data)), np.count_nonzero(np.isnan(data_clinical))))
nan_values = np.where(np.isnan(data_clinical))
for idx in range(nan_values[0].shape[0]):
    i = nan_values[0][idx]
    j = nan_values[1][idx]
    lab = labels[idx]
    tmp = data_clinical[labels == lab]
    tmp = tmp[:, j]
    tmp = tmp[~np.isnan(tmp)]
    data_clinical[i, j] = tmp.mean()
print("After NaN's markers {}, clinical {}".format(np.count_nonzero(
    np.isnan(data)), np.count_nonzero(np.isnan(data_clinical))))

# FIX CLINICAL
idx_height = features_clinical.index("a_height_mm")
idx_mass = features_clinical.index("a_bodymass_kg")
list_to_correct = ["01797_04110_20160503"]

for file_id in range(len(files_data.tolist())):
    if files_data[file_id].split("-")[0] in list_to_correct:
        print(data_clinical[file_id, idx_height], data_clinical[file_id, idx_mass], files_data[file_id])
        data_clinical[file_id, idx_height] = 1610
        data_clinical[file_id, idx_mass] = 42


# NORMALIZE
heights = data_clinical[:, idx_height]

# X & Y
print("X & Y: ", data[0, 1, 0, :], data[0, 12, 100, :])

# normalize y step 1
# normalize with height of the patient
data[:, :, :, 1] /= heights[:, None, None]
centers = []
for idx in range(data.shape[0]):
    # normalize x betwwn 0 and 1
    data[idx, :, :, 0] = (data[idx, :, :, 0] - data[idx, :, :, 0].min()) / \
        (data[idx, :, :, 0].max() - data[idx, :, :, 0].min())

    # normalize y step 2
    # find center and move to new center 0.5
    center = (0.5 - (data[idx, :, :, 1].max() + data[idx, :, :, 1].min()) / 2)
    data[idx, :, :, 1] += center
    centers.append(center)

    # make sure people walk in the same direction and normalize X and Y
    if data[idx, 10, 0, 0] > data[idx, 10, 100, 0]:
        data[idx, :, :, 0] = 1 - data[idx, :, :, 0]
        data[idx, :, :, 1] = 0.5 + (0.5 - data[idx, :, :, 1])

print("X & Y: ", data[0, 1, 0, :], data[0, 12, 100, :])

# Z
print("Z: ", data[0, 10, 0, :], data.shape, heights.shape)
data[:, :, :, 2] /= heights[:, None, None]
print("Z: ", data[0, 10, 0, :], data.shape, heights.shape)

# ADD velocity
data_vel = data[:, :, 1:, :] - data[:, :, :-1, :]
print(data_vel.shape, data.shape)
data = np.concatenate((data[:, :, 1:, :], data_vel), axis=3)
print(data.shape)


files_unique = [f_name.split("-")[0] for f_name in files_data]
files_unique = list(set(files_unique))
random.shuffle(files_unique)


content = "METADATA\n\nSEPARATE OVER VISITS (unique patient means same patient in 2 different visits = 2 unique patients)\n\n"
content += """INFORMATIONS ABOUT NORMALISATION
First we make sure every pathient walk in the same direction, if not we reverse the walk to fit same direction.

We used 3 different types of normalization (one for each coordinate):
    X: Corresponds to the distance, normalize between 0 and 1 --> 0 to the lhee and 1 to the right toe (rtoe)
    (data[idx, :, :, 0] - data[idx, :, :, 0].min()) / (data[idx, :, :, 0].max() - data[idx, :, :, 0].min())
        - Normalization: cycle coordinate 0 - min(cycle coordinate 0) / (max(cycle coordinate 0) - min(cycle coordinate 0))
        - Denormalization: normalized cord * (max(cycle coordinate 0) - min(cycle coordinate 0)) + min(cycle coordinate 0)

    Y: Corresponds to the movement of markers from left to right, patients do not necessarily walk in the center of the line (point 0).
    We normalized first with the height to have the right proportions of the body and then centered in 0.5
        - Normalization: 
            1) coordinate 1 / heights
            2) find min and max per cycle
            3) compute center : (0.5 - (max + min) / 2) # 0.5 is for new center
            4) move center to new center 0.5: data[idx, :, :, 1] += center
        - Denormalization: coordinate 2 * heights
            1) move back to old center: data[idx, :, :, 1] -= center
            2) coordinate 1 * heights

    Z: Corresponds to the height of the markers, normalized simply by dividing by the height of each patient.
        - Normalization: coordinate 2 / heights
        - Denormalization: coordinate 2 * heights
\n\n
"""

p = 0.7
n = len(files_unique)

content += "\np = {}\nnumber of unique patients = {}\n\n".format(p, n)

patients_train = files_unique[:int(p*n)]
patients_val = files_unique[int(p*n):int(((p+((1-p)/2))*n))]
patients_test = files_unique[int(((p+((1-p)/2))*n)):]


train_idx = [idx for idx in range(len(files_data))
             if files_data[idx].split("-")[0] in patients_train]
val_idx = [idx for idx in range(len(files_data))
           if files_data[idx].split("-")[0] in patients_val]
test_idx = [idx for idx in range(len(files_data))
            if files_data[idx].split("-")[0] in patients_test]

random.shuffle(train_idx)
random.shuffle(val_idx)
random.shuffle(test_idx)


data_train = data[train_idx]
data_val = data[val_idx]
data_test = data[test_idx]

clinical_train = data_clinical[train_idx]
clinical_val = data_clinical[val_idx]
clinical_test = data_clinical[test_idx]

heights_train = heights[train_idx]
heights_val = heights[val_idx]
heights_test = heights[test_idx]

centers = np.array(centers)
centers_train = centers[train_idx]
centers_val = centers[val_idx]
centers_test = centers[test_idx]

files_d = np.array(files_data)
train_files = files_d[train_idx]
val_files = files_d[val_idx]
test_files = files_d[test_idx]

labels = np.array(labels)
train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_labels = labels[test_idx]


content += "number of unique patients in train = {}\n".format(
    len(patients_train))
content += "number of unique patients in val = {}\n".format(len(patients_val))
content += "number of unique patients in test = {}\n\n".format(
    len(patients_test))


content += "Labels in train: {} (0)left, {} (1)right\n".format(
    np.count_nonzero(train_labels == 0), np.count_nonzero(train_labels == 1))
content += "Labels in val: {} (0)left, {} (1)right\n".format(
    np.count_nonzero(val_labels == 0), np.count_nonzero(val_labels == 1))
content += "Labels in test: {} (0)left, {} (1)right\n\n".format(
    np.count_nonzero(test_labels == 0), np.count_nonzero(test_labels == 1))

content += "\nDATA:\n"
content += "Train shape: {}\n".format(data_train.shape)
content += "Val shape: {}\n".format(data_val.shape)
content += "Test shape: {}\n".format(data_test.shape)

content += "\nCLINICAL:\n"
content += "Train shape: {}\n".format(clinical_train.shape)
content += "Val shape: {}\n".format(clinical_val.shape)
content += "Test shape: {}\n".format(clinical_test.shape)

content += "\n\npatients in train = {}\n".format(patients_train)
content += "patients in val = {}\n".format(patients_val)
content += "patients in test = {}\n\n".format(patients_test)

content += "\nmarkers : {}\n".format(markers_names_final)
content += "\nclinical features : {}\n".format(features_clinical)

content += "\nNan values in clinical replaced with mean for a given gol only for patients with same side affected\n"

content += "\nLABELS:\n"
content += "Train: {}\n".format(train_labels)
content += "Val: {}\n".format(val_labels)
content += "Test: {}\n".format(test_labels)


path_to_save = os.path.join(path_to_save, folder)

os.makedirs(path_to_save, exist_ok=True)

np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "motion", "train", dataset_name)), data_train)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "motion", "valid", dataset_name)), data_val)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "motion", "test", dataset_name)), data_test)

mean = clinical_train.mean(0)
std = clinical_train.std(0)
content += "\n\nMEAN FOR CLINICAL = {}".format(mean.tolist())
content += "\n\nSTD FOR CLINICAL = {}".format(std.tolist())

np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "clinical", "train", dataset_name)), (clinical_train - mean) / std)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "clinical", "valid", dataset_name)), (clinical_val - mean) / std)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "clinical", "test", dataset_name)), (clinical_test - mean) / std)

np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "files", "train", dataset_name)), train_files)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "files", "valid", dataset_name)), val_files)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "files", "test", dataset_name)), test_files)


np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "heights", "train", dataset_name)), heights_train)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "heights", "valid", dataset_name)), heights_val)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "heights", "test", dataset_name)), heights_test)

np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "centers", "train", dataset_name)), centers_train)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "centers", "valid", dataset_name)), centers_val)
np.save(os.path.join(path_to_save, "{}_{}_CP_hemiplegia_cycle_{}_markers.npy".format(
    "centers", "test", dataset_name)), centers_test)


with open(os.path.join(path_to_save, "metadat.txt"), "w") as file_:
    file_.write(content)
