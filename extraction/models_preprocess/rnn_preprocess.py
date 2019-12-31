import numpy as np
import os
import sys
import cv2
from utils import *
from pathlib import Path

# Construct dataset with fixed nb frames to 100
def construct_dataset (files_csv_path, y_csv_path, markers_path) :
    files_csv_path = load_csv(files_csv_path, dtype="U100", skiprows=0)
    y_csv_path = load_csv(y_csv_path, skiprows=0)
    batch = None
    y = None
    errors = 0
    for i in range(files_csv_path.shape[0]):
        try:
            file = files_csv_path[i].split('.')[0] + ".npy"
            data = np.load(markers_path + file)
    
            data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
            data = data.astype(float)

            res = cv2.resize(data, dsize=(100, data.shape[0]), interpolation=cv2.INTER_CUBIC)
            res = res[np.newaxis, :, :]
            
            label = convert_to_one_hot(y_csv_path[i])
    
            if batch is not None:
                batch = np.vstack((batch, res))
                y = np.vstack((y, label))
            else:
                batch = res
                y = label
    
        except:
            print("Problem with the file", file)
            errors += 1

    print("There were", errors, "errors")
    print("Batch dimensions : ", batch.shape)
    print("Label dimensions of the batch", y.shape)
    
    print("Nb instances in batch : ", batch.shape[0])
    print("Nb labels : ", y.shape[0])
    print("Nb different labels : ", y.shape[1])

    return batch, y

class Markers :
    def __init__(self, markers):
        self.markers = markers
        self.nb_frames = markers.shape[1]
        self.nb_markers = markers.shape[0]
        
    def getMarkers(self):
        return self.markers
    
    def setMarkers(self, markers):
        self.markers = markers


# Construct dataset with dynamic nb of frames
def construct_dataset_dynamic_frames (files_csv_path, y_csv_path, markers_path) :
    files_csv_path = load_csv(files_csv_path, dtype="U100", skiprows=0)
    y_csv_path = load_csv(y_csv_path, skiprows=0)
    batch = None
    y = None
    errors = 0
    for i in range(files_csv_path.shape[0]):
        try:
            file = files_csv_path[i].split('.')[0] + ".npy"
            data = np.load(markers_path + file)
            data = data.astype(float)
            data = np.vstack((np.vstack((data[:, :, 0], data[:, :, 1])), data[:, :, 2]))
            
            label = convert_to_one_hot(y_csv_path[i])
            
    
            if batch is not None:
                batch = np.vstack((batch, Markers(data)))
                y = np.vstack((y, label))
            else:
                batch = Markers(data)
                y = label                    
        except:
            print("Problem with the file", file)
            errors += 1

    print("There were", errors, "errors")
    print("Batch dimensions for the first row : ", batch[0][0].getMarkers().shape)
    print("Batch dimensions for the second row : ", batch[1][0].getMarkers().shape)
    print("Batch dimensions (array of object containing the markers) : ", batch.shape)
    print("Label dimensions of the batch", y.shape)
    print("Nb instances in batch : ", batch.shape[0])
    print("Nb labels : ", y.shape[0])
    print("Nb different labels : ", y.shape[1])

    return batch, y

# Normalize data with fixed number of frame
def normalize_data(X, mean=None, std=None, train=False) :
    if train:
        mean = np.mean(X, axis=(0, 2))
        X = X - mean[np.newaxis, :, np.newaxis]
            
        std = np.std(X, axis=(0, 2))
        X = X / std[np.newaxis, :, np.newaxis]
    
        return X, mean, std
    else :
        X = X - mean[np.newaxis, :, np.newaxis]
            
        X = X / std[np.newaxis, :, np.newaxis]
    
        return X


# Normalize data with number of frames dynamic
def normalize_dynamic_data(X, mean=None, std=None, train=False) :
    # Create zeros matrix with as many row as there are frames in all the dataset and with as many column as there are markers
    markers_values_all = None
    
    # Get all markers values and create a structure with it
    print("Create matrix of all markers stacked in nb_markers columns ...")
    for i in range(X.shape[0]):
        markers = X[i][0].getMarkers()
        
        if markers_values_all is not None:
            markers_values_all = np.vstack((markers_values_all, markers.T))
        else :
            markers_values_all = markers.T    
    print("Matrix of all markers created with shape : ", markers_values_all.shape)
    
    if train:
        mean = np.mean(markers_values_all, axis=0)
        std = np.std(markers_values_all, axis=0)
        
    print("Normalize the markers for all dataset ... :")
    for i in range(X.shape[0]):
        markers = X[i][0].getMarkers().T
        markers_normalized = (markers - mean) / std
        X[i][0].setMarkers(markers_normalized.T)
    print("Markers normalized")
    
    print("Check normalization with new mean and std (should give mean of 0 and std of 1) on training set ...")
    if train:
        markers_values_all= None
        for i in range(X.shape[0]):
            markers = X[i][0].getMarkers()

            if markers_values_all is not None:
                markers_values_all = np.vstack((markers_values_all, markers.T))
            else:
                markers_values_all = markers.T
        print("New mean : ", np.mean(markers_values_all, axis=0))
        print("New std : ", np.std(markers_values_all, axis=0))
    
    return X, mean, std
    


def prepare_and_extract () :
    # Project directory to make this code runnable on any windows system (to be changed on mac)
    project_dir = os.path.expanduser(os.path.dirname(Path(os.getcwd()).parent))

    # Get all directory names for cross validation
    dir_names = os.listdir(project_dir + "\\data\\cross_validated\\")
    dir_names = filter(lambda k: 'CV' in k, dir_names)

    for folder in dir_names:
        ################################################################################
        # Dynamic nb frames
        ################################################################################
        # Before starting, create directories to save the pre processed data if it doesn't already exist
        path = project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder
        if not os.path.exists(path):
            os.makedirs(path)

        # Train paths
        files_train_csv_path = project_dir + "\\data\\cross_validated\\" + folder + "\\files_train.csv"
        y_train_csv_path = project_dir + "\\data\\cross_validated\\" + folder + "\\y_train.csv"
        markers_train_path = project_dir + "\\data\\cross_validated\\" + folder + "\\markers\\train\\"

        # Test paths
        files_test_csv_path = project_dir + "\\data\\cross_validated\\" + folder + "\\files_test.csv"
        y_test_csv_path = project_dir + "\\data\\cross_validated\\" + folder + "\\y_test.csv"
        markers_test_path = project_dir + "\\data\\cross_validated\\" + folder + "\\markers\\test\\"

        # Dynamic nb frames
        # Get training data and labels
        x, y = construct_dataset_dynamic_frames(files_train_csv_path, y_train_csv_path, markers_train_path)

        # Normalize the data and get back their mean and std
        x_norm, mean, std = normalize_dynamic_data(x, train=True)

        # Save dataset and y to be able reload after
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\x_train_norm_dynamic", x_norm)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\y_train", y)

        # Get testing data and labels
        x, y = construct_dataset_dynamic_frames(files_test_csv_path, y_test_csv_path, markers_test_path)

        # Normalize the data with the training mean and std to apply the same normalization
        x_norm, _, _ = normalize_dynamic_data(x, mean, std, train=False)

        # Save dataset and y to be able reload after
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\x_test_norm_dynamic", x_norm)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\y_test", y)

        # Save training mean and std to be reused if the model will be in production
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\mean_train", mean)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\std_train", std)

        ################################################################################
        # Fixed nb frames
        ################################################################################
        # Before starting, create directories to save the pre processed data if it doesn't already exist
        path = project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder
        if not os.path.exists(path):
            os.makedirs(path)

        # Get training data and labels
        X, y = construct_dataset(files_train_csv_path, y_train_csv_path, markers_train_path)
        
        # Normalize the data and get back their mean and std
        X_norm, mean, std = normalize_data(X, train=True)
        
        # Save dataset and y to be able reload after
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\x_train_norm", X_norm)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\y_train", y)
        
        
        # Get testing data and labels 
        X, y = construct_dataset(files_test_csv_path, y_test_csv_path, markers_test_path)
        
        # Normalize the data with the training mean and std to apply the same normalization
        X_norm = normalize_data(X, mean, std, train=False)
        
        # Save dataset and y to be able reload after
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\x_test_norm", X_norm)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\y_test", y)

        # Save training mean and std to be reused if the model will be in production
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\mean_train", mean)
        np.save(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\std_train", std)


if __name__ == '__main__':
    prepare_and_extract()
