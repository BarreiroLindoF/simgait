import os
import random

import numpy as np

from utils import *

def split_folds(input_folder, output_folder, k=10, seed=13):
    metadata = "METADATA : \n\nNumber of k-folds = {}".format(k)
    os.makedirs(output_folder, exist_ok=True)

    files = load_csv(join_path(input_folder, 'files.csv'), dtype=str, skiprows=0)
    for f_idx in range(len(files)):
        name_splitted = files[f_idx].split('.')
        files[f_idx] = name_splitted[0] + '.' + name_splitted[1].upper()
    
    random.seed(seed)

    diagnostics = load_csv(join_path(input_folder, 'y.csv'), dtype=str, skiprows=0)
    dict_clean = {}
    for idx in range(len(diagnostics)):
        diag = ''.join(diagnostics[idx])
        if diag not in dict_clean.keys():
            dict_clean[diag] = [files[idx][:5]]
        else:
            dict_clean[diag].append(files[idx][:5])

    stats_patients = {}
    for key, value in dict_clean.items():
        dict_clean[key] = list(set(value))
        dict_clean[key].sort()
        random.shuffle(dict_clean[key])

        stats_patients[key] = len(dict_clean[key])

    metadata += "\n\nNumber of patients total: {}".format(np.sum(list(stats_patients.values())))
    metadata += "\n\tNormaux ({}): {}".format('0 0 0 0', stats_patients['0 0 0 0'])
    metadata += "\n\tCP_hemiplegia_left ({}): {}".format('1 0 0 0', stats_patients['1 0 0 0'])
    metadata += "\n\tCP_hemiplegia_right ({}): {}".format('0 1 0 0', stats_patients['0 1 0 0'])
    metadata += "\n\tCP_diplegia ({}): {}".format('1 1 0 0', stats_patients['1 1 0 0'])
    metadata += "\n\tITW_triceps ({}): {}".format('0 0 1 0', stats_patients['0 0 1 0'])
    metadata += "\n\tITW_soleus_triceps ({}): {}".format('0 0 0 1', stats_patients['0 0 0 1'])
    
    for key, value in stats_patients.items():
        stats_patients[key] = [value, value-(value//k), value//k]

    train_total = np.sum([value[1] for _, value in stats_patients.items()])
    test_total = np.sum([value[2] for _, value in stats_patients.items()])

    metadata += "\n\nNumber of patients in train : {}".format(train_total)
    metadata += "\n\tNormaux ({}): {}".format('0 0 0 0', stats_patients['0 0 0 0'][1])
    metadata += "\n\tCP_hemiplegia_left ({}): {}".format('1 0 0 0', stats_patients['1 0 0 0'][1])
    metadata += "\n\tCP_hemiplegia_right ({}): {}".format('0 1 0 0', stats_patients['0 1 0 0'][1])
    metadata += "\n\tCP_diplegia ({}): {}".format('1 1 0 0', stats_patients['1 1 0 0'][1])
    metadata += "\n\tITW_triceps ({}): {}".format('0 0 1 0', stats_patients['0 0 1 0'][1])
    metadata += "\n\tITW_soleus_triceps ({}): {}".format('0 0 0 1', stats_patients['0 0 0 1'][1])

    metadata += "\nNumber of patients in each test fold : {}".format(test_total)
    metadata += "\n\tNormaux ({}): {}".format('0 0 0 0', stats_patients['0 0 0 0'][2])
    metadata += "\n\tCP_hemiplegia_left ({}): {}".format('1 0 0 0', stats_patients['1 0 0 0'][2])
    metadata += "\n\tCP_hemiplegia_right ({}): {}".format('0 1 0 0', stats_patients['0 1 0 0'][2])
    metadata += "\n\tCP_diplegia ({}): {}".format('1 1 0 0', stats_patients['1 1 0 0'][2])
    metadata += "\n\tITW_triceps ({}): {}".format('0 0 1 0', stats_patients['0 0 1 0'][2])
    metadata += "\n\tITW_soleus_triceps ({}): {}".format('0 0 0 1', stats_patients['0 0 0 1'][2])

    content_input_folder = os.listdir(input_folder)

    for val_k in range(1, k+1):
        cv_folder = "CV_{}".format(val_k)
        os.makedirs(join_path(output_folder, cv_folder), exist_ok=True)
        metadata += "\n\nCV {} : ".format(val_k)

        train_list = []
        test_list = []

        for key, value in dict_clean.items():
            test_list_tmp = dict_clean[key][stats_patients[key][2]*(val_k-1):stats_patients[key][2]*(val_k)]
            train_list += list(set(dict_clean[key]) - set(test_list_tmp))
            test_list += test_list_tmp

        #test_list = list_clean[n_ppatients_fold*(val_k-1):n_ppatients_fold*(val_k)]
        #train_list = list(set(list_clean) - set(test_list))

        train_idx = [i for i in range(len(files)) if files[i][:5] in train_list]
        test_idx = [i for i in range(len(files)) if files[i][:5] in test_list]

        random.shuffle(train_idx)
        random.shuffle(test_idx)

        metadata += "\nNumber of samples in train : {}".format(train_idx.count(True))
        metadata += "\nNumber of samples in test : {}".format(test_idx.count(True))
        for content in content_input_folder:
            if os.path.isfile(join_path(input_folder, content)) and content.split('.')[-1] == 'csv':
                if content == 'features_labels.csv':
                    features_labels = load_csv(join_path(input_folder, content), dtype=str, skiprows=0)
                    save_csv(join_path(output_folder, cv_folder, content), features_labels)
                elif content == 'files.csv':  
                    save_csv(join_path(output_folder, cv_folder, 'files_train'), files[train_idx])
                    save_csv(join_path(output_folder, cv_folder, 'files_test'), files[test_idx])
                else:
                    data = load_csv(join_path(input_folder, content), skiprows=0)
                    name = content.split('.')[0]
                    save_csv(join_path(output_folder, cv_folder, '{}_train'.format(name)), data[train_idx].astype(str))
                    save_csv(join_path(output_folder, cv_folder, '{}_test'.format(name)), data[test_idx].astype(str))

            elif os.path.isdir(join_path(input_folder, content)):
                os.makedirs(join_path(output_folder, cv_folder, content), exist_ok=True)
                if content == 'angles_aligned':
                    def stack_save_data(side, metadata):
                        os.makedirs(join_path(output_folder, cv_folder, content, side), exist_ok=True)
                        files_alignned = get_files(join_path(input_folder, content, side))
                        
                        files_list = files.tolist()
                        y_data = load_csv(join_path(input_folder, 'y.csv'), dtype=str, skiprows=0)
                        train_data = []
                        train_y = []
                        test_data = []
                        test_y  = []                 
                        for f in files_alignned:
                            name = f.split('.')[0]
                            if f[:5] in train_list:
                                train_data.append(np.load(join_path(input_folder, content, side, f)))
                                for idx_size in range(train_data[-1].shape[0]):
                                    train_y.append(y_data[files_list.index("{}.C3D".format(name))].tolist())
                            if f[:5] in test_list:
                                test_data.append(np.load(join_path(input_folder, content, side, f)))
                                for idx_size in range(test_data[-1].shape[0]):
                                    test_y.append(y_data[files_list.index("{}.C3D".format(name))].tolist())
                        metadata += "\nNumber of samples in train (angles aligned {}): {}".format(side, len(train_y))
                        metadata += "\nNumber of samples in test (angles aligned {}): {}".format(side, len(test_y))
                        np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_train"), np.concatenate(train_data))
                        np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_test"), np.concatenate(test_data))
                        save_csv(join_path(output_folder, cv_folder, content, side, "y_train"), train_y)
                        save_csv(join_path(output_folder, cv_folder, content, side, "y_test"), test_y)
                        # data_stacked = [np.load(join_path(input_folder, content, side, f)) for f in files_alignned if f[:5] in train_list]
                        # np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_train"), np.concatenate(data_stacked))
                        # # test
                        # data_stacked = [np.load(join_path(input_folder, content, side, f)) for f in files_alignned if f[:5] in test_list]
                        # np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_test"), np.concatenate(data_stacked))
                        return metadata
                    metadata = stack_save_data('Left', metadata)
                    metadata = stack_save_data('Right', metadata)
                elif content == 'markers_aligned':
                    def stack_save_data(side, metadata):
                        os.makedirs(join_path(output_folder, cv_folder, content, side), exist_ok=True)
                        files_alignned = get_files(join_path(input_folder, content, side))
                        
                        files_list = files.tolist()
                        y_data = load_csv(join_path(input_folder, 'y.csv'), dtype=str, skiprows=0)
                        train_data = []
                        train_y = []
                        test_data = []
                        test_y  = []                 
                        for f in files_alignned:
                            name = f.split('.')[0]
                            if f[:5] in train_list:
                                train_data.append(np.load(join_path(input_folder, content, side, f)))
                                for idx_size in range(train_data[-1].shape[0]):
                                    train_y.append(y_data[files_list.index("{}.C3D".format(name))].tolist())
                            if f[:5] in test_list:
                                test_data.append(np.load(join_path(input_folder, content, side, f)))
                                for idx_size in range(test_data[-1].shape[0]):
                                    test_y.append(y_data[files_list.index("{}.C3D".format(name))].tolist())
                        metadata += "\nNumber of samples in train (markers aligned {}): {}".format(side, len(train_y))
                        metadata += "\nNumber of samples in test (markers aligned {}): {}".format(side, len(test_y))
                        np.save(join_path(output_folder, cv_folder, content, side, "markers_aligned_train"), np.concatenate(train_data))
                        np.save(join_path(output_folder, cv_folder, content, side, "markers_aligned_test"), np.concatenate(test_data))
                        save_csv(join_path(output_folder, cv_folder, content, side, "y_train"), train_y)
                        save_csv(join_path(output_folder, cv_folder, content, side, "y_test"), test_y)
                        # data_stacked = [np.load(join_path(input_folder, content, side, f)) for f in files_alignned if f[:5] in train_list]
                        # np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_train"), np.concatenate(data_stacked))
                        # # test
                        # data_stacked = [np.load(join_path(input_folder, content, side, f)) for f in files_alignned if f[:5] in test_list]
                        # np.save(join_path(output_folder, cv_folder, content, side, "angles_aligned_test"), np.concatenate(data_stacked))
                        return metadata
                    metadata = stack_save_data('Left', metadata)
                    metadata = stack_save_data('Right', metadata)
                else:
                    files_angles = get_files(join_path(input_folder, content))
                    os.makedirs(join_path(output_folder, cv_folder, content, 'train'), exist_ok=True)
                    os.makedirs(join_path(output_folder, cv_folder, content, 'test'), exist_ok=True)
                    for f in files_angles:
                        data = np.load(join_path(input_folder, content, f))
                        if f[:5] in train_list:
                            np.save(join_path(output_folder, cv_folder, content, 'train', f), data)
                        elif f[:5] in test_list:
                            np.save(join_path(output_folder, cv_folder, content, 'test', f), data)

        file_write(join_path(output_folder,'metadata.txt'), metadata)


if __name__ == "__main__":

    project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

    cleaned_raw_data = project_dir + '\\data\\cleaned'

    split_folds(cleaned_raw_data, project_dir + '\\data\\cross_validated', k=4)