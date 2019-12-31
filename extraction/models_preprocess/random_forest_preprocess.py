import numpy as np
import os
from pathlib import Path
from utils import *

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(Path(os.getcwd()).parent))

# Directory paths
examinations_path = project_dir + "\\data\\extracted\\examination.csv"


# Construct dataset with fixed nb frames to 100
def construct_dataset(files_csv_path):
    data = load_csv(examinations_path, delimiter=", ", dtype="U100", skiprows=0)

    # Get the labels and the index of each one just in case
    labels_col_names = data[0, :]

    # Delete headers first row
    data = np.delete(data, 0, axis=0)

    # Get labels (the last column of the dataset)
    y = data[:, -1]

    # Delete y from the dataset to separate data and labels in two differents variables
    data = np.delete(data, -1, axis=1)

    # Categorize column "sexe" with 0 for men and 1 for women
    sex_col_idx = np.where(labels_col_names[:] == 'sex')[0]
    for i in range(data.shape[0]):
        if data[i, sex_col_idx] == 'F':
            data[i, sex_col_idx] = 1
        else:
            data[i, sex_col_idx] = 0

    data = data.astype(float)

    # Make a total of NaN per instance and delete some of them that have more than 15% of NaN. For the rest, get the mean of all and replace NaN with it
    # Go through lines
    nb_nan_row = np.zeros(data.shape[0])
    nb_nan_col = np.zeros(data.shape[1])
    means_col = np.zeros(data.shape[1])
    for i in range(data.shape[0]):
        # Go through columns
        for j in range(data.shape[1]):
            if data[i, j] == ' NaN' or data[i, j] == 'NaN' or np.isnan(data[i, j]):
                nb_nan_row[i] += 1
                nb_nan_col[j] += 1
            else:
                # Manually calculate the mean (because matrix with differents types and NaN)
                means_col[j] += data[i, j]

    nb_val_per_col = np.repeat(data.shape[0], data.shape[1]) - nb_nan_col

    means_col /= nb_val_per_col

    max_nan_ratio = 0.15
    ratio_nan = nb_nan_row / data.shape[1]
    # Rows that have more than 15% of NAN in it
    rows_to_del = np.where(ratio_nan > max_nan_ratio)

    # Delete these lines
    data = np.delete(data, rows_to_del[0], axis=0)
    y = np.delete(y, rows_to_del[0])

    # Replace all nan value by the correspondant mean
    for i in range(data.shape[0]):
        # Go through columns
        for j in range(data.shape[1]):
            if data[i, j] == ' NaN' or data[i, j] == 'NaN' or np.isnan(data[i, j]):
                data[i, j] = means_col[j]

    print("Labels shape : ", y.shape)
    print("Data shape : ", data.shape)
    return data, y


def prepare_and_extract():
    # Get training data and labels
    X, y = construct_dataset(examinations_path)

    # Create directory to save if it doesn't exist
    path = project_dir + "\\data\\models_prepared\\random_forest_formated"
    if not os.path.exists(path):
        os.makedirs(path)

    # Save pre processed data
    np.save(project_dir + "\\data\\models_prepared\\random_forest_formated\\x", X)
    np.save(project_dir + "\\data\\models_prepared\\random_forest_formated\\y", y)


if __name__ == '__main__':
    prepare_and_extract()
