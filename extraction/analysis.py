import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.expanduser("~/workspace/bitbucket/joao/code/tools"))

try:
    from my_utils import get_files, join_path, load_csv
except:
    raise

ANGLES = [
    "RAnkleAngles",
    "LAnkleAngles",
    "RHipAngles",
    "LHipAngles",
    "RPelvisAngles",
    "LKneeAngles",
    "RKneeAngles",
    "LPelvisAngles",
]
PATH_TO_CLEANED_DATA = (
    "C:\\Temp\\SimGait\\researchprojectongait\\extraction\\data_cleaned_CP2"
)
PATH_TO_EXTRACTED_DATA = (
    "C:\\Temp\\SimGait\\researchprojectongait\\extraction\\data_extracted_CP"
)
PATH_TO_ALIGNED_DATA = "C:\\Temp\\SimGait\\researchprojectongait\\extraction\\data_extracted_CP\\data_cleaned_CP2\\angles_aligned"


def get_clinical_data(path, rows_max_nan=30, cols_max_nan=50, verbose=True, plots=False):
    list_to_remove = ["c_force_exthallucisl", "c_force_triceps", "c_select_hipadductor", "c_select_peroneuslongusbrevis", "c_select_tibialisanterior", "c_spastashworth_hipflexor", "c_spastashworth_kneeextensor", "c_spastashworth_kneeflexor"]

    data_clinical = load_csv(join_path(path, "examination.csv"), dtype="str", skiprows=0)
    files_list = load_csv(join_path(path, "files.csv"), dtype="str", skiprows=1)
    files_list = np.array([file_.split("\\")[-1].split('.')[0] for file_ in files_list])

    data_clinical = data_clinical[:, 2:]

    features = data_clinical[0]
    data_clinical = data_clinical[1:]
    if verbose:
        print("Clinical data shape: {}".format(data_clinical.shape))

    data_clinical[data_clinical[:, 1] == " M", 1] = 1
    data_clinical[data_clinical[:, 1] == " F", 1] = 0
    data_clinical = data_clinical.astype(float)

    nan_in_cols = np.array(
        [
            np.count_nonzero(np.isnan(data_clinical[:, idx]))
            for idx in range(data_clinical.shape[1])
        ]
    )
    if verbose:
        display_clinical_stats(data_clinical)

    if plots:
        nan_in_cols.sort()
        plt.plot(nan_in_cols)
        plt.show()

    cols_idx_keep = [True if val < cols_max_nan else False for val in nan_in_cols]


    nan_in_rows = np.array(
        [
            np.count_nonzero(np.isnan(data_clinical[idx, :]))
            for idx in range(data_clinical.shape[0])
        ]
    )

    data_clinical = data_clinical[:, cols_idx_keep]
    features = features[cols_idx_keep]

    rows_idx_keep = [True if val < rows_max_nan else False for val in nan_in_rows]
    data_clinical = data_clinical[rows_idx_keep, :]
    files = files_list[rows_idx_keep]

    
    id_feat = []
    sides = ["left", "right", "l", "r"]
    for ele in features:
        splited = [e for e in ele.split("_") if e not in sides]
        joined = "_".join(splited).replace(" ", "")
        if joined in list_to_remove:
            id_feat.append(False)
        else:
            id_feat.append(True)
            
    data_clinical = data_clinical[:, id_feat]
    features = features[id_feat]

    features = [feat.replace(" ", "") for feat in features]

    if verbose:
        display_clinical_stats(data_clinical)
    return data_clinical, features, files


def display_clinical_stats(data_clinical):
    nan_in_rows = np.array(
        [
            np.count_nonzero(np.isnan(data_clinical[idx, :]))
            for idx in range(data_clinical.shape[0])
        ]
    )
    nan_in_cols = np.array(
        [
            np.count_nonzero(np.isnan(data_clinical[:, idx]))
            for idx in range(data_clinical.shape[1])
        ]
    )
    print()
    print()
    print(nan_in_rows, len(nan_in_rows))
    print("NaNs in rows:")
    print("Total number of NaNs = {}".format(nan_in_rows.sum()))
    print("Max number of NaNs = {}".format(nan_in_rows.max()))
    print("Number of rows with NaNs = {}".format((nan_in_rows > 0).sum()))
    print("Number of rows with zero NaNs = {}".format((nan_in_rows == 0).sum()))
    print()
    print(nan_in_cols)
    print("NaNs in cols:")
    print("Total number of NaNs = {}".format(nan_in_cols.sum()))
    print("Max number of NaNs = {}".format(nan_in_cols.max()))
    print("Number of rows with NaNs = {}".format((nan_in_cols > 0).sum()))
    print("Number of rows with zero NaNs = {}".format((nan_in_cols == 0).sum()))


def get_aligned_data(path_to_data, path_to_folder, side):
    # get files
    path = join_path(path_to_data, side)
    list_files = get_files(path, "npy")

    files_names = load_csv(
        join_path(path_to_folder, "files.csv"), dtype="str", skiprows=0
    )
    files_names = [name.split("\\")[-1].split('.')[0] for name in files_names]
    affected_side = load_csv(join_path(path_to_folder, "y.csv"), skiprows=0)

    # get data
    motion_data_all = []
    labels = []
    files = []
    for f_name in list_files:
        full_path = join_path(path, f_name)
        data = np.load(full_path)
        motion_data_all.append(data)
        idx = files_names.index(f_name.split(".")[0])
        labels += [affected_side[idx]] * data.shape[0]
        files += [f_name.split(".")[0]] * data.shape[0]

    # concatenate data into a numpy array
    data_all = np.concatenate(motion_data_all, axis=0)
    print(data_all.shape)
    return data_all, np.array(labels), files


def remove_other_side(data_left, data_right):
    """Keeps the left angles for left cuts and right angles for right cuts."""
    idx_left = [idx for idx in range(len(ANGLES)) if ANGLES[idx][0] == "L"]
    idx_right = [idx for idx in range(len(ANGLES)) if ANGLES[idx][0] == "R"]

    print(data_left.shape)
    print(data_right.shape)
    # for now use only first coordinate
    data_left = data_left[:, idx_left, :, 0]
    data_right = data_right[:, idx_right, :, 0]

    data_left = np.swapaxes(data_left, 0, 1)
    data_right = np.swapaxes(data_right, 0, 1)

    print(data_left.shape)
    print(data_right.shape)

    return data_left, data_right


def show_all(side_affected, data_left, data_right):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Side affected : {}".format(side_affected))
    std_ratio = 1

    axs[0, 0].plot(data_left[:, 0, :].mean(0), label="left")
    axs[0, 0].fill_between(
        range(101),
        data_left[:, 0, :].mean(0) + std_ratio * data_left[:, 0, :].std(0),
        data_left[:, 0, :].mean(0) - std_ratio * data_left[:, 0, :].std(0),
        alpha=0.2,
    )
    axs[0, 0].plot(data_right[:, 0, :].mean(0), label="right")
    axs[0, 0].fill_between(
        range(101),
        data_right[:, 0, :].mean(0) + std_ratio * data_right[:, 0, :].std(0),
        data_right[:, 0, :].mean(0) - std_ratio * data_right[:, 0, :].std(0),
        alpha=0.2,
    )
    axs[0, 0].set_title("Ankle")

    axs[0, 1].plot(data_left[:, 1, :].mean(0), label="left")
    axs[0, 1].fill_between(
        range(101),
        data_left[:, 1, :].mean(0) + std_ratio * data_left[:, 1, :].std(0),
        data_left[:, 1, :].mean(0) - std_ratio * data_left[:, 1, :].std(0),
        alpha=0.2,
    )
    axs[0, 1].plot(data_right[:, 1, :].mean(0), label="right")
    axs[0, 1].fill_between(
        range(101),
        data_right[:, 1, :].mean(0) + std_ratio * data_right[:, 1, :].std(0),
        data_right[:, 1, :].mean(0) - std_ratio * data_right[:, 1, :].std(0),
        alpha=0.2,
    )
    axs[0, 1].set_title("Hip")

    axs[1, 0].plot(data_left[:, 2, :].mean(0), label="left")
    axs[1, 0].fill_between(
        range(101),
        data_left[:, 2, :].mean(0) + std_ratio * data_left[:, 2, :].std(0),
        data_left[:, 2, :].mean(0) - std_ratio * data_left[:, 2, :].std(0),
        alpha=0.2,
    )
    axs[1, 0].plot(data_right[:, 3, :].mean(0), label="right")
    axs[1, 0].fill_between(
        range(101),
        data_right[:, 3, :].mean(0) + std_ratio * data_right[:, 3, :].std(0),
        data_right[:, 3, :].mean(0) - std_ratio * data_right[:, 3, :].std(0),
        alpha=0.2,
    )
    axs[1, 0].set_title("Knee")

    axs[1, 1].plot(data_left[:, 3, :].mean(0), label="left")
    axs[1, 1].fill_between(
        range(101),
        data_left[:, 3, :].mean(0) + std_ratio * data_left[:, 0, :].std(0),
        data_left[:, 3, :].mean(0) - std_ratio * data_left[:, 0, :].std(0),
        alpha=0.2,
    )
    axs[1, 1].plot(data_right[:, 2, :].mean(0), label="right")
    axs[1, 1].fill_between(
        range(101),
        data_right[:, 2, :].mean(0) + std_ratio * data_right[:, 2, :].std(0),
        data_right[:, 2, :].mean(0) - std_ratio * data_right[:, 2, :].std(0),
        alpha=0.2,
    )
    axs[1, 1].set_title("Pelvis")

    for ax in axs.flat:
        ax.set(xlabel="x-label", ylabel="y-label")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend()
    plt.show()

def show_difference(data1, label1, data2, label2):
    plt.clf()
    plt.cla()

    std_ratio = 1
    alpha = 0.2

    plt.plot(data1.mean(0), label=label1)
    plt.fill_between(
        range(101),
        data1.mean(0) + std_ratio * data1.std(0),
        data1.mean(0) - std_ratio * data1.std(0),
        alpha=alpha,
    )

    plt.plot(data2.mean(0), label=label2)
    plt.fill_between(
        range(101),
        data2.mean(0) + std_ratio * data2.std(0),
        data2.mean(0) - std_ratio * data2.std(0),
        alpha=alpha,
    )
    plt.legend()
    #plt.show()
    name = "{}_{}.pdf".format(label1, label2)
    plt.savefig(name, format="pdf", dpi=1200, bbox_inches="tight", transparent=True, pad_inches=0.05)


if __name__ == "__main__":
    # get data
    data_left, labels_left, _ = get_aligned_data(PATH_TO_ALIGNED_DATA, PATH_TO_CLEANED_DATA, "Left")
    data_right, labels_right, _ = get_aligned_data(PATH_TO_ALIGNED_DATA, PATH_TO_CLEANED_DATA, "Right")
    data_clinical = get_clinical_data(PATH_TO_EXTRACTED_DATA)

    # data_left for left cut left angles, data right for right cut right angles
    data_left, data_right = remove_other_side(data_left, data_right)

    data_left_left = data_left[labels_left == 0, :, :]
    data_right_left = data_right[labels_right == 0, :, :]

    data_left_right = data_left[labels_left == 1, :, :]
    data_right_right = data_right[labels_right == 1, :, :]

    plots = True
    if plots:
        show_all("Left", data_left_left, data_right_left)
        show_all("Right", data_left_right, data_right_right)

        # left affected on left side vs right affected on right side
        show_difference(
            data_left_left[:, 0, :], "LALS", data_right_right[:, 0, :], "RARS"
        )

        # left affected on left side vs right affected on left side
        show_difference(
            data_left_left[:, 0, :], "LALS", data_left_right[:, 0, :], "RALS"
        )

        # left affected on left side vs left affected on right side
        show_difference(
            data_left_left[:, 0, :], "LALS", data_right_left[:, 0, :], "LARS"
        )

        # right affected on right side vs right affected on left side
        show_difference(
            data_right_right[:, 0, :], "RARS", data_left_right[:, 0, :], "RALS"
        )

        # right affected on right side vs left affected on right side
        show_difference(
            data_right_right[:, 0, :], "RARS", data_right_left[:, 0, :], "LARS"
        )

        # right affected on left side vs left affected on right side
        show_difference(
            data_left_right[:, 0, :], "RALS", data_right_left[:, 0, :], "LARS"
        )

    #get_clinical_data()
