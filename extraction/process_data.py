import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.append(os.path.expanduser("~/workspace/bitbucket/joao/code/tools"))

try:
    from my_utils import *
except:
    raise

list_to_remove_left = ['00775_01593_20100614-GBNNN-VDEF-06.npy', '00419_00763_20090624-GBNNN-VDEF-11.npy', '00067_00239_20080305-GBNNN-VDEF-10.npy', '00501_01493_20100503-GBNNN-VDEF-09.npy', '00478_00866_20091111-GBNNN-VDEF-08.npy', '00032_00085_20071205-GBNNN-VDEF-05.npy', '00237_00379_20080730-GBNNN-VDEF-15.npy', '00419_00763_20090624-GBNNN-VDEF-12.npy', '01814_02780_20121010-GBNNN-VDEF-23.npy', '00984_01573_20100602-GBNNN-VDEF-11.npy', '00478_00866_20091111-GBNNN-VDEF-07.npy',
                       '00501_01493_20100503-GBNNN-VDEF-11.npy', '00501_01493_20100503-GBNNN-VDEF-06.npy', '00134_02027_20110309-GBNNN-VDEF-12.npy', '00032_00085_20071205-GBNNN-VDEF-04.npy', '00419_00763_20090624-GBNNN-VDEF-15.npy', '00052_00841_20091021-GBNNN-VDEF-09.npy', '00478_00866_20091111-GBNNN-VDEF-13.npy', '00237_00379_20080730-GBNNN-VDEF-11.npy', '00052_00841_20091021-GBNNN-VDEF-11.npy', '00478_00866_20091111-GBNNN-VDEF-14.npy', '01745_02618_20120507-GBNNN-VDEF-10.npy', '00192_00469_20080825-GBNNN-VDEF-19.npy']
list_to_remove_right = ['00118_00598_20081105-GBNNN-VDEF-12.npy', '00567_01123_20100331-GBNNN-VDEF-08.npy', '00419_00763_20090624-GBNNN-VDEF-11.npy', '00025_00299_20080319-GBNNN-VDEF-18.npy', '00501_01493_20100503-GBNNN-VDEF-09.npy', '00237_00379_20080730-GBNNN-VDEF-13.npy', '00478_00866_20091111-GBNNN-VDEF-08.npy', '00032_00085_20071205-GBNNN-VDEF-05.npy', '00237_00379_20080730-GBNNN-VDEF-15.npy', '00419_00763_20090624-GBNNN-VDEF-12.npy', '00984_01573_20100602-GBNNN-VDEF-11.npy', '00052_00841_20091021-GBNNN-VDEF-13.npy', '01870_02866_20121203-GBNNN-VDEF-14.npy',
                        '00501_01493_20100503-GBNNN-VDEF-11.npy', '01969_03046_20130527-GBNNN-VDEF-10.npy', '00419_00763_20090624-GBNNN-VDEF-15.npy', '00192_00469_20080825-GBNNN-VDEF-13.npy', '00984_01573_20100602-GBNNN-VDEF-13.npy', '00607_01920_20110117-GBNNN-VDEF-05.npy', '00052_00841_20091021-GBNNN-VDEF-09.npy', '00237_00379_20080730-GBNNN-VDEF-11.npy', '01971_03070_20130624-GBNNN-VDEF-23.npy', '00478_00866_20091111-GBNNN-VDEF-14.npy', '00025_00299_20080319-GBNNN-VDEF-06.npy', '00567_01123_20100331-GBNNN-VDEF-16.npy', '00048_00647_20081222-GBNNN-VDEF-13.npy']


def align_and_save_data(data, events, output_folder, file_, type_data="angles"):
    def align_save_side(side):
        os.makedirs(join_path(output_folder, "{}_aligned".format(
            type_data), side), exist_ok=True)
        idx_list = events[
            np.logical_and(
                events[:, 1] == " {}".format(
                    side), events[:, 2] == " FootStrike"
            ),
            0,
        ].astype(int)
        cycle_data = []
        for idx in range(len(idx_list) - 1):
            cycle_angle = []
            for n_angle in range(data.shape[0]):
                x = np.linspace(0, 101, num=idx_list[idx + 1] - idx_list[idx])
                cycle_cord = []
                for cord in range(3):
                    f = interp1d(
                        x,
                        data[n_angle, idx_list[idx]: idx_list[idx + 1], cord],
                        kind="cubic",
                    )
                    cycle_cord.append(f(np.arange(0, 101)))
                cycle_angle.append(np.array(cycle_cord).T)
            cycle_data.append(np.array(cycle_angle))
        if len(cycle_data):
            np.save(
                join_path(output_folder, "{}_aligned".format(
                    type_data), side, file_),
                np.array(cycle_data),
            )

    if type_data == "markers":
        if file_ not in list_to_remove_left:
            align_save_side("Left")
        else:
            print(file_)
        if file_ not in list_to_remove_right:
            align_save_side("Right")
        else:
            print(file_)
    else:
        align_save_side("Left")
        align_save_side("Right")


def get_examination_data(path, idx_keep, th_rows, th_cols, plots):
    examination_data = load_csv(path, dtype=str)

    # replace strings of sex by 1 for males and 0 for females
    examination_data[examination_data[:, 1] == " M", 1] = 1
    examination_data[examination_data[:, 1] == " F", 1] = 0

    examination_data = examination_data.astype(float)
    print("exa :", examination_data.shape)
    examination_data = examination_data[idx_keep]
    print("exa :", examination_data.shape)
    # remove also some idx because of missing values in examination data
    rows = []
    for idx in range(examination_data.shape[0]):
        rows.append(np.count_nonzero(np.isnan(examination_data[idx, :])))

    if plots:
        hist = Counter(rows)
        plt.bar(hist.keys(), hist.values())
        # plt.axhline(th_rows, color='r')
        plt.xlabel("number of nan features per sample")
        plt.ylabel("number of samples")
        plt.show()

    if th_rows is None:
        th_rows = np.mean(rows)

    idx_rows_keep = np.array(rows) < th_rows
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
        # plt.axhline(th_cols, color='r')
        plt.xlabel("number of nan samples per feature")
        plt.ylabel("number of features")
        plt.show()

    idx_cols_keep = np.array(cols) < th_cols
    examination_data = examination_data[:, idx_cols_keep]

    nan_idx = np.argwhere(np.isnan(examination_data))

    # replace nan values
    for idx in nan_idx:
        examination_data[idx[0], idx[1]] = np.nanmean(
            examination_data[:, idx[1]])

    idx_std_zero = np.std(examination_data, axis=0) != 0.0
    examination_data = examination_data[:, idx_std_zero]

    i = 0
    for idx in range(len(idx_cols_keep)):
        if idx_cols_keep[idx]:
            idx_cols_keep[idx] = idx_std_zero[i]
            i += 1

    print("saa : ", np.mean(examination_data, axis=0))
    examination_data = (examination_data - np.mean(examination_data, axis=0)) / np.std(
        examination_data, axis=0
    )

    idx_rows_keep = idx_keep
    return examination_data, idx_rows_keep, idx_cols_keep


def get_angles_data(input_folder, output_folder, files_keep, type_data="angles", align=True):
    files_keep_clean = [file_name.split("\\")[-1].split('.')[0] for file_name in files_keep]

    # print(files_keep_clean)

    files_angles = get_files(join_path(input_folder, type_data))

    if align:
        files_events = get_files(join_path(input_folder, "events"))

    os.makedirs(join_path(output_folder, type_data), exist_ok=True)
    for file_ in files_angles:


        if file_.split(".")[0] in files_keep_clean:

            data = np.load(join_path(input_folder, type_data,
                                     file_), allow_pickle=True)
            if len(data.shape) == 3:
                if np.count_nonzero(np.isnan(data)):
                    continue
            else:
                continue
            np.save(join_path(output_folder, type_data, file_), data)
            if align:
                events = load_csv(
                    join_path(
                        input_folder, "events", "{}.csv".format(
                            file_.split(".")[0])
                    ),
                    dtype=str,
                )
                align_and_save_data(
                    data, events, output_folder, file_, type_data=type_data)


def main(
    raw_data_folder,
    output_folder="data",
    keep_pathology="all",
    th_rows=22,
    th_cols=30,
    plots=True,
):
    files = get_files(raw_data_folder, extension="csv")

    diag_file = "diagnostic.csv"
    if diag_file not in files:
        raise Exception(
            "ERROR : You must have an {} file if your keep_pathology is not all".format(
                diag_file
            )
        )

    diag_list = load_csv(join_path(raw_data_folder, diag_file), dtype=str)

    idx_keep = [True] * diag_list.shape[0]
    if keep_pathology != "all":
        idx_keep = None
        if type(keep_pathology) is list:
            idx_keep = []
            for diag_idx in range(len(diag_list)):
                if diag_list[diag_idx] in keep_pathology:
                    idx_keep.append(True)
                else:
                    idx_keep.append(False)
        else:
            idx_keep = diag_list[:] == keep_pathology

    print("asdfafds", idx_keep[:10])

    examination_data, idx_rows_keep, idx_cols_keep = get_examination_data(
        join_path(raw_data_folder, "examination.csv"),
        idx_keep,
        th_rows=th_rows,
        th_cols=th_cols,
        plots=plots,
    )

    side_data = load_csv(
        join_path(raw_data_folder, 'affected_side.csv'), dtype=str)
    print(side_data.shape, side_data[:10])
    print(idx_rows_keep[:10])
    side_data = side_data[idx_rows_keep]

    for idx in range(side_data.shape[0]):
        if side_data[idx].lower() in ['left', 'gauche']:
            side_data[idx] = 0
        elif side_data[idx].lower() in ['right', 'droit']:
            side_data[idx] = 1
        else:
            side_data[idx] = 2

    print("1")
    files = load_csv(join_path(raw_data_folder, "files.csv"), dtype=str)
    files = files[idx_rows_keep]
    print("2")
    features_labels = load_csv(
        join_path(raw_data_folder, "examination.csv"), dtype=str, skiprows=0
    )[0]
    features_labels = features_labels[idx_cols_keep]
    print("3")
    os.makedirs(output_folder, exist_ok=True)

    # remove artefacts due tu values near 0
    data1, data2 = examination_data.copy(), examination_data.copy()
    data1[examination_data < 1e-14] = 0
    data2[examination_data > -1e-14] = 0
    examination_data = data1 + data2
    save_csv(join_path(output_folder, "examination.csv"),
             examination_data.astype(str))
    save_csv(join_path(output_folder, "y.csv"), side_data.astype(str))
    save_csv(join_path(output_folder, "files.csv"), files)
    save_csv(join_path(output_folder, "features_labels.csv"), features_labels)
    angles_data = get_angles_data(
        raw_data_folder, output_folder, files, type_data="angles", align=True)
    markers_data = get_angles_data(
        raw_data_folder, output_folder, files, type_data="markers", align=True)

    print("1")


if __name__ == "__main__":
    currPath = os.path.dirname(os.getcwd())
    raw_data_folder = currPath + "\\data\\extracted\\CP"
    keep_pathology = ["CP_Spastic_Uni_Hemiplegia", "CP_Spastic_Diplegia"]
    # keep_pathology = ["CP_Spastic_Uni_Hemiplegia"]
    main(
        raw_data_folder,
        output_folder=currPath + "\\data\\cleaned\\CP",
        keep_pathology=keep_pathology,
        plots=False,
    )
