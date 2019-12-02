import numpy as np
import os
from pathlib import Path


data_path = str(Path(os.path.dirname(os.getcwd())).parent) + "\data"
data_cleaned_path = data_path + "\cleaned"
data_extracted_path = data_path + "\extracted"

CP_path_dataset = data_cleaned_path + "\CP\dataset\CP_hemiplegia_cycle_armless_markers"
CP_path_extracted = data_extracted_path + "\CP"

ITW_path_dataset = data_cleaned_path + "\ITW\dataset\CP_hemiplegia_cycle_armless_markers"
ITW_path_extracted = data_extracted_path + "\ITW"

def main ():    
    """
    Train, validation and test dataset are constructed like : 
        (N,C,F,A) where N = Nb instances, C = NB sensors, F = NB frames, A = NB axis (x,y,z without no changes and x,y,z with velocity added)
        Here in the train set we have a shape of (541, 15, 100, 6). So 541 instances with 15 sensors in 100 frames with x,y,z values and x,y,z values with velocity added
        
    """
    train_CP = np.load(CP_path_dataset + "/motion_train_CP_hemiplegia_cycle_armless_markers.npy")
    validation_CP = np.load(CP_path_dataset + "/motion_valid_CP_hemiplegia_cycle_armless_markers.npy")
    test_CP = np.load(CP_path_dataset + "/motion_test_CP_hemiplegia_cycle_armless_markers.npy")
    
    train_CP_files = np.load(CP_path_dataset + "/files_train_CP_hemiplegia_cycle_armless_markers.npy")
    validation_CP_files = np.load(CP_path_dataset + "/files_valid_CP_hemiplegia_cycle_armless_markers.npy")
    test_CP_files = np.load(CP_path_dataset + "/files_test_CP_hemiplegia_cycle_armless_markers.npy")

    
    pathologies_each_instance_CP = np.loadtxt(CP_path_extracted + "\\files_and_pathologies.csv", delimiter=",", dtype=np.str)
    pathologies_CP = np.loadtxt(CP_path_extracted + "\pathologies.csv" ,delimiter=",", dtype=np.str)
    
    
    print(train_CP_files.shape)
    print(validation_CP_files.shape)
    print(test_CP_files.shape)
    
    print(pathologies_CP.shape)
    print(pathologies_CP)
    
    print(pathologies_each_instance_CP.shape)
    print(pathologies_each_instance_CP)
    
    print(train_CP.shape)
    print(validation_CP.shape)
    print(test_CP.shape)
    
    labels = []
    for i in range (train_CP_files.shape[0]):
        pathology_idx = np.where(pathologies_each_instance_CP[:,0] == train_CP_files[i])[0]
        #labels.append(pathologies_each_instance_CP[pathology_idx,1])
        labels.append(pathologies_each_instance_CP[pathology_idx,1][0])
    labels = np.asarray(labels)
    print(labels.shape)
    
main()