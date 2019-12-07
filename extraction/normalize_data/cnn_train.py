import numpy as np
from utils import *
import cv2

files_train_csv = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\files_train.csv"
y_train_csv = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\y_train.csv"
markers_train = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\markers\\train\\"

files_train_csv = load_csv(files_train_csv, dtype="U100", skiprows=0)
y_train_csv = load_csv(y_train_csv, skiprows=0)
batch = None
y = None
errors = 0
for i in range(files_train_csv.shape[0]):
    try:
        file = files_train_csv[i].split('.')[0] + ".npy"
        data = np.load(markers_train + file)
        # print("Before formatting", data[0, :5, 0])

        data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
        # print(data.shape)
        # print("After formatting:", data[0, :5])
        # print(data.shape)
        # dsize = (width, height)
        res = cv2.resize(data, dsize=(100, data.shape[0]), interpolation=cv2.INTER_CUBIC)
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

mean = np.mean(batch, axis=(0, 2))
# print(mean)
batch = batch - mean[np.newaxis, :, np.newaxis]
# new_mean = np.mean(batch, axis=(0, 2))
# print(new_mean)

std = np.std(batch, axis=(0, 2))
# print(std)
batch = batch / std[np.newaxis, :, np.newaxis]
# new_std = np.std(batch, axis=(0, 2))
# print(new_std)

print("Saving")
np.save("1d_X_train", batch)
np.save("1d_y_train", y)
np.save("mean_train", mean)
np.save("std_train", std)


files_test_csv = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\files_test.csv"
y_test_csv = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\y_test.csv"
markers_test = "D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\cross_val\\CV_4\\markers\\test\\"

files_test_csv = load_csv(files_test_csv, dtype="U100", skiprows=0)
y_test_csv = load_csv(y_test_csv, skiprows=0)
batch = None
y = None
errors = 0
for i in range(files_test_csv.shape[0]):
    try:
        file = files_test_csv[i].split('.')[0] + ".npy"
        data = np.load(markers_test + file)
        # print("Before formatting", data[0, :5, 0])

        data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
        # print(data.shape)
        # print("After formatting:", data[0, :5])
        # print(data.shape)
        # dsize = (width, height)
        res = cv2.resize(data, dsize=(100, data.shape[0]), interpolation=cv2.INTER_CUBIC)
        res = res[np.newaxis, :, :]

        label = convert_to_one_hot(y_test_csv[i])

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

# print(mean)
batch = batch - mean[np.newaxis, :, np.newaxis]
# new_mean = np.mean(batch, axis=(0, 2))
# print(new_mean)

# std = np.std(batch, axis=(0, 2))
# print(std)
batch = batch / std[np.newaxis, :, np.newaxis]
# new_std = np.std(batch, axis=(0, 2))
# print(new_std)


np.save("1d_X_test", batch)
np.save("1d_y_test", y)

