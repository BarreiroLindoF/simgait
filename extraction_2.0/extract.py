import os

import numpy as np
from numpy import savetxt
import pandas as pd 

try:
    from btk import btk
except:
    raise


class Extracter:
    def __init__(self, dataset_path, output_folder):
        self.dataset_path = dataset_path
        self.output_folder = output_folder
        
        self.btk_reader = btk.btkAcquisitionFileReader()
        self.metadata_reader = None
        self.files_name = np.asarray(os.listdir(self.dataset_path))
    
    def readPoints(self, acq):
        sensors = []
        #Iterate through each sensor data (Nb frame, NbAxis)
        for sensor in btk.Iterate(acq.GetPoints()):
            sensors.append(sensor.GetValues())
            print(sensor.GetValues().shape)
            print(sensor.GetLabel())
        print(np.array(sensors).shape)
        
    #Read some c3d files with the list of path given in parameters and return the Acquisition object with the filename
    def read_files(self):
        list_acq = []
        #Iterate over each file
        for idx_file in range(self.files_name.shape[0]):
            #Instantiate a reader with btk library
            reader = btk.btkAcquisitionFileReader()
            #Store the acquisition file from the c3d file
            reader.SetFilename(self.dataset_path + self.files_name[idx_file])
            reader.Update()
            acq = reader.GetOutput()
            list_acq.append(acq)
        return list_acq


currPath = os.path.dirname(os.getcwd())
patho_path = "CP\\CP_Gait_1.0\\"

# Define where the data to extract are
dataset_path = os.path.expanduser(currPath + "\\data\\raw\\" + patho_path)
# Define where the data extracted will go
output_folder =  currPath + "\data\extracted\CP"

# Instantiate the extracter
extracter = Extracter(dataset_path, output_folder)

list_acq = extracter.read_files()

for j in range(len(list_acq)):
    #Get back the different points for each acquisition file in all axis except "axis_to_del" param
    extracter.readPoints(list_acq[j])

# Save all in some data structure such as csv, npy, txt
extracter.save_all()




