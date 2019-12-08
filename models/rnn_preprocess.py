import numpy as np
from utils import *
import cv2

files_train_csv = "C:\\Users\\lucas\\Desktop\\gaitmasteris\\data\\extracted\\cross_val\\CV_1\\files_train.csv"
y_train_csv = "C:\\Users\\lucas\\Desktop\\gaitmasteris\\data\\extracted\\cross_val\\CV_1\\y_train.csv"
markers_train = "C:\\Users\\lucas\\Desktop\\gaitmasteris\\data\\extracted\\cross_val\\CV_1\\markers\\train\\"

def construct_dataset (files_csv_path, y_csv_path, markers_path) :
    files_csv_path = load_csv(files_csv_path, dtype="U100", skiprows=0)
    y_csv_path = load_csv(y_csv_path, skiprows=0)
    batch = []
    y = []
    errors = 0
    for i in range(files_csv_path.shape[0]):
        try:
            file = files_csv_path[i].split('.')[0] + ".npy"
            data = np.load(markers_path + file)
    
            data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
                
            label = convert_to_one_hot(y_csv_path[i])
    
            batch.append(data)
            y.append(label)
    
        except:
            print("Problem with the file", file)
            errors += 1

        print("There were", errors, "errors")
        print("Nb instances in the batch", len(batch))
        print("Nb labels of the batch", len(y))
        
        print("Dimensions of first instance in the batch : ", batch[0])
        print("Dimensions of first label in the batch : ", y[0])
        
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
        
        print(batch.shape)

    return batch


construct_dataset(files_train_csv, y_train_csv, markers_train)