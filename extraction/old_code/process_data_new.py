import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import *

def align_and_save_data(data, events, output_folder, file_):
    def align_save_side(side):
        os.makedirs(join_path(output_folder, 'angles_aligned', side), exist_ok=True)
        idx_list = events[np.logical_and(events[:, 1]==' {}'.format(side), events[:, 2]==' FootStrike'), 0].astype(int)
        cycle_data = []
        for idx in range(len(idx_list)-1):
            cycle_angle = []
            for n_angle in range(data.shape[0]):
                x = np.linspace(0, 101, num=idx_list[idx+1]-idx_list[idx])
                cycle_cord = []
                for cord in range(3):
                    f = interp1d(x, data[n_angle, idx_list[idx]:idx_list[idx+1], cord], kind='cubic')
                    cycle_cord.append(f(np.arange(0, 101)))
                cycle_angle.append(np.array(cycle_cord).T)
            cycle_data.append(np.array(cycle_angle))
        if len(cycle_data):
            np.save(join_path(output_folder, 'angles_aligned', side, file_), np.array(cycle_data))
    align_save_side('Left')
    align_save_side('Right')

def get_examination_data(path, idx_keep, th_rows, th_cols, plots):
    examination_data = load_csv(path, dtype=str)

    # replace strings of sex by 1 for males and 0 for females
    examination_data[examination_data[:, 1] == ' M', 1] = 1
    examination_data[examination_data[:, 1] == ' F', 1] = 0

    examination_data = examination_data.astype(float)
    print('exa :', examination_data.shape)
    examination_data = examination_data[idx_keep]
    print('exa :', examination_data.shape)
    # remove also some idx because of missing values in examination data
    rows = []
    for idx in range(examination_data.shape[0]):
        rows.append(np.count_nonzero(np.isnan(examination_data[idx, :])))

    if plots:
        hist = Counter(rows)
        plt.bar(hist.keys(), hist.values())
        #plt.axhline(th_rows, color='r')
        plt.xlabel('number of nan features per sample')
        plt.ylabel('number of samples')
        plt.show()

    if th_rows is None:
        th_rows = np.mean(rows)
    
    idx_rows_keep = (np.array(rows) < th_rows)
    print(idx_rows_keep[:10])
    examination_data = examination_data[idx_rows_keep, :]

    i = 0
    for idx in range(len(idx_keep)):
        if idx_keep[idx]:
            idx_keep[idx] = idx_rows_keep[i]
            i += 1
    print(idx_keep[:10])

    cols = []
    for idx in range(examination_data.shape[1]):
        cols.append(np.count_nonzero(np.isnan(examination_data[:, idx])))

    if th_cols is None:
        th_cols = np.mean(cols)

    if plots:
        hist = Counter(cols)
        plt.bar(hist.keys(), hist.values())
        #plt.axhline(th_cols, color='r')
        plt.xlabel('number of nan samples per feature')
        plt.ylabel('number of features')
        plt.show()

    idx_cols_keep = (np.array(cols) < th_cols)
    examination_data = examination_data[:, idx_cols_keep]

    nan_idx = np.argwhere(np.isnan(examination_data))

    # replace nan values
    for idx in nan_idx:
        examination_data[idx[0], idx[1]] = np.nanmean(examination_data[:, idx[1]])

    idx_std_zero = (np.std(examination_data, axis=0) != 0.)
    examination_data = examination_data[:, idx_std_zero]
    
    i = 0
    for idx in range(len(idx_cols_keep)):
        if idx_cols_keep[idx]:
            idx_cols_keep[idx] = idx_std_zero[i]
            i += 1

    examination_data = (examination_data - np.mean(examination_data, axis=0))/np.std(examination_data, axis=0)

    idx_rows_keep = idx_keep
    return examination_data, idx_rows_keep, idx_cols_keep

def get_angles_data(input_folder, output_folder, files_keep, align=True):
    files_keep_clean = [file_name.split('.')[0] for file_name in files_keep]
    files_angles = get_files(join_path(input_folder, 'angles'))
    if align:
        files_events = get_files(join_path(input_folder, 'events'))
    
    os.makedirs(join_path(output_folder, 'angles'), exist_ok=True)
    for file_ in files_angles:
        print(file_)
        if file_.split('.')[0] in files_keep_clean:
            data = np.load(join_path(input_folder, 'angles', file_))   
            # if np.count_nonzero(np.isnan(data)):
            #     print("Found non zero")
            #     continue

            np.save(join_path(output_folder, 'angles', file_), data)
            if align:
                events = load_csv(join_path(input_folder, 'events', "{}.csv".format(file_.split('.')[0])), dtype=str)
                align_and_save_data(data, events, output_folder, file_)



def main(raw_data_folder, output_folder='data', keep_pathology='all', th_rows=22, th_cols=30, plots=True):
    files = get_files(raw_data_folder, extension='csv')

    diag_file = 'diagnostic.csv'
    if diag_file not in files:
        raise Exception("ERROR : You must have an {} file if your keep_pathology is not all".format(diag_file))
    
    diag_list = load_csv(join_path(raw_data_folder, diag_file), dtype=str)

    idx_keep = [True] * diag_list.shape[0]
    if keep_pathology != 'all':
        idx_keep = None
        if type(keep_pathology) is list:
            idx_keep = []
            for diag_idx in range(len(diag_list)):
                if diag_list[diag_idx] in keep_pathology:
                    idx_keep.append(True)
                else:
                    idx_keep.append(False)
        else:
            idx_keep = (diag_list[:] == keep_pathology)

    print('asdfafds', idx_keep[:10])

    # examination_data, idx_rows_keep, idx_cols_keep = get_examination_data(join_path(raw_data_folder, 'examination.csv'), idx_keep, th_rows=th_rows, th_cols=th_cols, plots=plots)

    side_data = load_csv(join_path(raw_data_folder, 'affected_side.csv'), dtype=str, skiprows=1)
    #print(side_data.shape, side_data[:10])
    #print(idx_rows_keep[:10])
    #side_data = side_data[idx_rows_keep]
    side_data = side_data.tolist()[:]

    # for idx in range(side_data.shape[0]):
    #     if side_data[idx].lower() in ['left', 'gauche']:
    #         side_data[idx] = 0
    #     elif side_data[idx].lower() in ['right', 'droit']:
    #         side_data[idx] = 1
    #     else:
    #         side_data[idx] = 2

    for idx in range(len(side_data)):
        if side_data[idx].lower() in ['left', 'gauche']:
            side_data[idx] = ['0', '1', '0', '0']
        elif side_data[idx].lower() in ['right', 'droit']:
            side_data[idx] = ['1', '0', '0', '0']
        elif side_data[idx].lower() in ['normal']:
            side_data[idx] = ['0', '0', '0', '0']
        elif side_data[idx].upper() in ['ITW_RETRACTION_TRICEPS']:
            side_data[idx] = ['0', '0', '1', '0']
        elif side_data[idx].upper() in ['ITW_RETRACTION_SOL_TRICEPS']:
            side_data[idx] = ['0', '0', '0', '1']
        else:
            side_data[idx] = ['1', '1', '0', '0']

    print('1')
    files = load_csv(join_path(raw_data_folder, 'files.csv'), dtype=str)
    #files = files[idx_rows_keep]
    print('2')
    # features_labels = load_csv(join_path(raw_data_folder, 'examination.csv'), dtype=str, skiprows=0)[0]
    # features_labels = features_labels[idx_cols_keep]
    print('3')
    os.makedirs(output_folder, exist_ok=True)
    # save_csv(join_path(output_folder, 'examination.csv'), examination_data.astype(str))
    #save_csv(join_path(output_folder, 'y.csv'), side_data.astype(str))
    save_csv(join_path(output_folder, 'y.csv'), side_data)
    save_csv(join_path(output_folder, 'files.csv'), files)
    #save_csv(join_path(output_folder, 'features_labels.csv'), features_labels)
    angles_data = get_angles_data(raw_data_folder, output_folder, files, align=True)
    print('1')

if __name__ == "__main__":
    raw_data_folder = 'temp_folders/data_extracted_pablo'
    keep_pathology = 'CP_Spastic_Uni_Hemiplegia'
    keep_pathology = ['CP_Spastic_Uni_Hemiplegia', 'CP_Spastic_Bi_Diplegia']
    keep_pathology = ['CP_Spastic_Uni_Hemiplegia', 
                      'CP_Spastic_Bi_Diplegia',
                      'Idiopathic toe walker',
                      'Healthy']
    main(raw_data_folder, output_folder='temp_folders/data_cleaned_pablo', keep_pathology=keep_pathology, plots=False)
