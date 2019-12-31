import numpy as np
import os
from pathlib import Path
from utils import *
import cv2

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(Path(os.getcwd()).parent))

# Differents path to save the data pre processed (depending on which model it will deserve)
path_width_100_m = project_dir + "\\data\\models_prepared\\cnn_formated\\Width100MarkersOnly\\"
path_width_200_m = project_dir + "\\data\\models_prepared\\cnn_formated\\Width200MarkersOnly\\"
path_width_200_m_a = project_dir + "\\data\\models_prepared\\cnn_formated\\Width200MarkersAngles\\"

# Array containing all paths
paths_to_save = [path_width_100_m, path_width_200_m, path_width_200_m_a]

# Array containing True if the model will use markers + angles and False if just markers
is_markers_angles = [False, False, True]

# Different width sizes
widths = [100, 200, 200]

############################################################################################################################################################

def construct_dataset(width, files_path, y_path, markers_path, angles_path=None):
    files_train_csv = load_csv(files_path, dtype="U100", skiprows=0)
    y_train_csv = load_csv(y_path, skiprows=0)
    batch = None
    y = None
    errors = 0

    for i in range(files_train_csv.shape[0]):
        try:
            # Split paths stored to get the filename and load it
            file = files_train_csv[i].split('.')[0] + ".npy"
            data = np.load(markers_path + file)

            data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))

            # If there is angles path, we will stack them with the markers
            if angles_path is not None:
                angles = np.load(angles_path + file)
                angles = np.vstack((np.vstack((angles[:, :, 0], angles[:, :, 1])), angles[:, :, 2]))

                data = np.vstack((data, angles))

            res = cv2.resize(data, dsize=(width, data.shape[0]), interpolation=cv2.INTER_CUBIC)
            res = res[np.newaxis, :, :]

            label = convert_to_one_hot(y_train_csv[i])

            if batch is not None:
                batch = np.vstack((batch, res))
                y = np.vstack((y, label))
            else:
                batch = res
                y = label

        except:
            # print("Problem with the file", file)
            errors += 1
    print("There were", errors, "errors")
    print("Size of the batch", batch.shape)
    print("Size of the labels", y.shape)

    return batch, y

def normalize_data(x, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(x, axis=(0, 2))
        std = np.std(x, axis=(0, 2))

    x = x - mean[np.newaxis, :, np.newaxis]
    x = x / std[np.newaxis, :, np.newaxis]

    return x, mean, std

def save(x, y, filename, path, mean=None, std=None, ):
    np.save(path + "1d_X_" + filename, x)
    np.save(path + "1d_y_" + filename, y)

    if mean is not None and std is not None:
        np.save(path + "mean_" + filename, mean)
        np.save(path + "std_" + filename, std)


if __name__ == '__main__':
    # Get all directory names for cross validation
    dir_names = os.listdir(project_dir + "\\data\\cross_validated\\")
    dir_names = filter(lambda k: 'CV' in k, dir_names)

    for folder in dir_names:
        # Training data set path for the cross validation (data to preprocess)
        files_train_csv = project_dir + "\\data\\cross_validated\\" + folder + "\\files_train.csv"
        y_train_csv = project_dir + "\\data\\cross_validated\\" + folder + "\\y_train.csv"
        markers_train = project_dir + "\\data\\cross_validated\\"+ folder + "\\markers\\train\\"
        angles_train = project_dir + "\\data\\cross_validated\\" + folder +"\\angles\\train\\"

        # Testing data set path for the cross validation (data to pre process)
        files_test_csv = project_dir + "\\data\\cross_validated\\" + folder + "\\files_test.csv"
        y_test_csv = project_dir + "\\data\\cross_validated\\" + folder + "\\y_test.csv"
        markers_test = project_dir + "\\data\\cross_validated\\" + folder + "\\markers\\test\\"
        angles_test = project_dir + "\\data\\cross_validated\\" + folder + "\\angles\\test\\"

        # Pre processing for different models and saving the results in different paths
        for j in range(len(paths_to_save)):
            print("Constructing training data set...")
            # If markers + angles needed for this model, give the angles path in param
            if is_markers_angles[j]:
                # Construct the train set and get the labels
                x_train, y_train = construct_dataset(widths[j], files_train_csv, y_train_csv, markers_train, angles_train)
            else:
                # Construct the train set and get the labels
                x_train, y_train = construct_dataset(widths[j], files_train_csv, y_train_csv, markers_train)

            print(x_train.shape)
            print("Normalizing training data set...")
            # Normalize the training set and get the mean and the std to apply the same transformation on test set
            x_train_norm, mean_train, std_train = normalize_data(x_train)

            print("Constructing testing data set...")
            # If markers + angles needed for this model, give the angles path in param
            if is_markers_angles[j]:
                # Construct the test set and get the labels
                x_test, y_test = construct_dataset(widths[j], files_test_csv, y_test_csv, markers_test, angles_test)
            else:
                # Construct the test set and get the labels
                x_test, y_test = construct_dataset(widths[j], files_test_csv, y_test_csv, markers_test)

            print("Normalizing testing data set...")
            # Normalize the testing set with the mean and the std of the training set
            x_test_norm, _, _ = normalize_data(x_test, mean_train, std_train)

            print("Saving all in : ", paths_to_save[j])
            # Save all to reuse it later
            path = paths_to_save[j] + folder
            if not os.path.exists(path):
                os.mkdir(path)
            save(x_train_norm, y_train, 'train', path + "\\", mean_train, std_train)
            save(x_test_norm, y_test, 'test', path + "\\")


################################################################################################

#
# files_test_csv = load_csv(files_test_csv, dtype="U100", skiprows=0)
# y_test_csv = load_csv(y_test_csv, skiprows=0)
# batch = None
# y = None
# errors = 0
# for i in range(files_test_csv.shape[0]):
#     try:
#         file = files_test_csv[i].split('.')[0] + ".npy"
#         data = np.load(markers_test + file)
#         # print("Before formatting", data[0, :5, 0])
#
#         data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
#
#         if is_markers_angle:
#             angles = np.load(angles_test + file)
#             angles = np.vstack((np.vstack((angles[:, :, 0], angles[:, :, 1])), angles[:, :, 2]))
#
#             data = np.vstack((data, angles))
#         # print(data.shape)
#         # print("After formatting:", data[0, :5])
#         # print(data.shape)
#         # dsize = (width, height)
#         res = cv2.resize(data, dsize=(width, data.shape[0]), interpolation=cv2.INTER_CUBIC)
#         res = res[np.newaxis, :, :]
#
#         label = convert_to_one_hot(y_test_csv[i])
#
#         if batch is not None:
#             batch = np.vstack((batch, res))
#             y = np.vstack((y, label))
#         else:
#             batch = res
#             y = label
#
#
#     except:
#         # print("Problem with the file", file)
#         errors += 1
# print("There were", errors, "errors")
# print("Size of the batch", batch.shape)
# print("Size of the labels", y.shape)
#
# # print(mean)
# batch = batch - mean[np.newaxis, :, np.newaxis]
# # new_mean = np.mean(batch, axis=(0, 2))
# # print(new_mean)
#
# # std = np.std(batch, axis=(0, 2))
# # print(std)
# batch = batch / std[np.newaxis, :, np.newaxis]
# # new_std = np.std(batch, axis=(0, 2))
# # print(new_std)
#
#
# np.save("1d_X_test", batch)
# np.save("1d_y_test", y)
