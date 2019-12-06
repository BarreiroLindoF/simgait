import os
import sys
from random import shuffle

import numpy as np

sys.path.append(os.path.expanduser("~/workspace/bitbucket/joao/code/tools"))

try:
    from my_utils import *
except:
    raise


def slice_motion_data(
    motion_data,
    velocities_data,
    context_data,
    examination,
    num_frames_per_slice=49,
    file_name="",
    overlap=True,
):
    slices = list()
    velocities_slices = list()
    context_slices = list()
    files_slices = list()
    num_frames = float(motion_data.shape[1])
    examination_all = []

    n_iter = int(np.floor(num_frames / num_frames_per_slice))
    if overlap:
        n_iter = int(num_frames - num_frames_per_slice)

    for slice_idx in range(n_iter):
        if overlap:
            start_idx = slice_idx
            end_idx = slice_idx + num_frames_per_slice
        else:
            start_idx = slice_idx * num_frames_per_slice
            end_idx = (slice_idx + 1) * num_frames_per_slice
        slices.append(motion_data[:, start_idx:end_idx, :])
        velocities_slices.append(velocities_data[:, start_idx:end_idx, :])
        context_slices.append(context_data[start_idx:end_idx])
        files_slices.append(file_name)

        examination_all.append()

    return slices, velocities_slices, context_slices, files_slices


def normalize_all(data_all, min_val=None, max_val=None):
    """Normalize globally to [-1, 1]"""
    if not min_val:
        min_val = data_all.min()
    if not max_val:
        max_val = data_all.max()
    data_all = (data_all - min_val) * 2 / (max_val - min_val) - 1
    return data_all, min_val, max_val


class data_spliter:
    def __init__(
        self, exp_name, exp_out, suffix, suffix_end_name, n_splits=3, display=True
    ):
        self.exp_name = exp_name
        self.exp_name_out = exp_out
        self.suffix = suffix
        self.suffix_end_name = suffix_end_name

        self.n_splits = n_splits

        self.display = display

        self.files = load_csv(
            exp_name.split("\\")[0] + "\files.csv", dtype="str", skiprows=0
        ).tolist()
        self.examination = load_csv(
            exp_name.split("\\")[0] + "\\examination.csv", skiprows=0
        )

        self.metadata = ""

    def getData(
        self,
        path,
        type,
        list_files,
        array=None,
        joints=38,
        VELOCITY=True,
        min_vals=None,
        max_vals=None,
        min_val_vel=None,
        max_val_vel=None,
        frames=50,
        overlap=True,
    ):
        data_all = []
        examination_all = []

        motion_data_all = list()
        velocities_all = list()
        context_all = []
        files_slice_all = []

        VELOCITY = False
        meanstd = True

        minmax = False
        if max_vals is None:
            minmax = True

        minmax = False
        min_vals, max_vals = None, None

        if array is not None:
            motion_data_all.append(array)
        for f_name in list_files:
            fname = os.path.join(path, f_name)
            # fname = os.path.join(os.path.join(path, type), f_name)
            data = np.load(fname)

            name_file = fname.split("\\")[-1][:-4] + ".C3D"
            id_name_file = self.files.index(name_file)

            # data = np.swapaxes(data, 0, 1)
            # if VELOCITY:
            #     velocities_all.append(data[:, 1:, :] - data[:, :-1, :])
            #     data = data[:, 1:, :]
            # if minmax:
            #     if max_vals is None:
            #         max_vals = data.max(1)
            #         min_vals = data.min(1)
            #     else:
            #         max_vals = np.maximum(max_vals, data.max(1))
            #         min_vals = np.minimum(min_vals, data.min(1))

            motion_data_all.append(data)
            examination_all += [self.examination[id_name_file]] * data.shape[0]

            files_slice_all += [fname.split("\\")[-1]] * data.shape[0]
            # context_all.append(np.arange(data.shape[1])+1)

        # if minmax:
        #     max_vals = np.expand_dims(max_vals, 1)
        #     min_vals = np.expand_dims(min_vals, 1)

        # slices_all = list()
        # context_slices_all = list()
        # files_slice_all = list()
        # velocities_slices_all = list()

        # for idx in range(len(motion_data_all)):
        #     data = motion_data_all[idx]

        # if not meanstd:
        #     data = (data - min_vals) * 2 / (max_vals - min_vals) - 1

        # data[np.isnan(data)] = 0

        # slices, velocities_slices, context_slices, files_slice = slice_motion_data(data, velocities_all[idx], context_all[idx], frames, list_files[idx], overlap=overlap)
        # slices_all.extend(slices)
        # velocities_slices_all.extend(velocities_slices)
        # context_slices_all.extend(context_slices)
        # files_slice_all.extend(files_slice)

        # print('slices_all : ', len(slices_all))

        data_all = np.concatenate(motion_data_all, axis=0)
        exam_all = np.array(examination_all)

        print(exam_all.shape)
        # velocities = np.array(velocities_slices_all, dtype=np.float32)
        # data_all_context = np.array(context_slices_all, dtype=np.float32)
        # files_slice_all = np.array(files_slice_all)

        if meanstd:
            if max_vals is None:
                mean = np.expand_dims(
                    np.expand_dims(np.mean(data_all, axis=(0, 2)), 0), 2
                )
                std = (
                    np.expand_dims(np.expand_dims(np.std(data_all, axis=(0, 2)), 0), 2)
                    + 1e-10
                )
            else:
                mean = min_vals
                std = max_vals
        data_all = (data_all - mean) / std

        if VELOCITY:
            velocities, min_val_vel, max_val_vel = normalize_all(
                velocities, min_val_vel, max_val_vel
            )
            data_all = np.concatenate((data_all, velocities), axis=3)

        # print(np.array(files_slice_all))

        if meanstd:
            min_vals = mean
            max_vals = std
        return (
            data_all,
            exam_all,
            min_vals,
            max_vals,
            None,
            None,
            None,
            np.array(files_slice_all),
        )  # , min_val_vel, max_val_vel, data_all_context, files_slice_all

    def split(self, length=50, overlap=True):
        self.metadata = ""
        for j in range(self.n_splits):
            self.metadata += "\n\nData " + str(j) + " :"

            mypath = self.exp_name

            # print(mypath)

            mypath_tmp = mypath  # os.path.join(mypath, 'angles')

            suffix = "{}_{}_{}".format(self.suffix, self.suffix_end_name, j)

            list_clean = [
                f[:5]
                for f in os.listdir(mypath_tmp)
                if os.path.isfile(os.path.join(mypath_tmp, f))
            ]
            list_clean = list(set(list_clean))
            shuffle(list_clean)

            # print(list_clean)

            p = 0.76
            train_len = int(len(list_clean) * p)
            if (len(list_clean) - train_len) % 2 == 1:
                train_len -= 1
            val_len = (len(list_clean) - train_len) // 2
            test_len = len(list_clean) - train_len - val_len

            train_list = list_clean[:train_len]
            val_list = list_clean[train_len : train_len + val_len]
            test_list = list_clean[train_len + val_len :]

            self.metadata += "\nTrain list of patients :" + str(train_list)
            self.metadata += "\nVal list of patients :" + str(val_list)
            self.metadata += "\nTest list of patients :" + str(test_list)

            train_list = [
                f
                for f in os.listdir(mypath_tmp)
                if os.path.isfile(os.path.join(mypath_tmp, f)) and f[:5] in train_list
            ]
            val_list = [
                f
                for f in os.listdir(mypath_tmp)
                if os.path.isfile(os.path.join(mypath_tmp, f)) and f[:5] in val_list
            ]
            test_list = [
                f
                for f in os.listdir(mypath_tmp)
                if os.path.isfile(os.path.join(mypath_tmp, f)) and f[:5] in test_list
            ]

            self.metadata += "\n\nTrain number of patients :" + str(train_len)
            self.metadata += "\nVal number of patients :" + str(val_len)
            self.metadata += "\nTest number of patients :" + str(test_len)
            self.metadata += "\n\nTrain number of files :" + str(len(train_list))
            self.metadata += "\nVal number of files :" + str(len(val_list))
            self.metadata += "\nTest number of files :" + str(len(test_list))

            # for type_name in ['angles', 'markers']:
            type_name = "angles"
            min_vals, max_vals, min_val_vel, max_val_vel = None, None, None, None
            data_train, exam_train, min_vals, max_vals, min_val_vel, max_val_vel, context_train, files_train = self.getData(
                path=mypath,
                type=type_name,
                list_files=train_list,
                frames=length,
                overlap=overlap,
            )
            data_valid, exam_valid, _, _, _, _, context_valid, files_valid = self.getData(
                path=mypath,
                type=type_name,
                list_files=val_list,
                min_vals=min_vals,
                max_vals=max_vals,
                min_val_vel=min_val_vel,
                max_val_vel=max_val_vel,
                frames=length,
                overlap=overlap,
            )
            data_test, exam_test, _, _, _, _, context_test, files_test = self.getData(
                path=mypath,
                type=type_name,
                list_files=test_list,
                min_vals=min_vals,
                max_vals=max_vals,
                min_val_vel=min_val_vel,
                max_val_vel=max_val_vel,
                frames=length * 2,
                overlap=overlap,
            )

            ind_train = np.arange(data_train.shape[0])
            ind_valid = np.arange(data_valid.shape[0])
            ind_test = np.arange(data_test.shape[0])

            np.random.shuffle(ind_train)
            np.random.shuffle(ind_valid)
            np.random.shuffle(ind_test)

            data_train = data_train[ind_train, :, :, :]
            data_valid = data_valid[ind_valid, :, :, :]
            data_test = data_test[ind_test, :, :, :]

            # context_train = context_train[ind_train, :]
            # context_valid = context_valid[ind_valid, :]
            # context_test = context_test[ind_test, :]

            files_train = files_train[ind_train]
            files_valid = files_valid[ind_valid]
            files_test = files_test[ind_test]

            exam_train = exam_train[ind_train]
            exam_valid = exam_valid[ind_valid]
            exam_test = exam_test[ind_test]

            suffix_final = "{}_{}".format(suffix, type_name)
            TARGET_PATH = os.path.join(self.exp_name_out, suffix_final)
            os.makedirs(TARGET_PATH, exist_ok=True)

            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH), "mean_" + suffix_final + ".npy"
                ),
                min_vals,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH), "std_" + suffix_final + ".npy"
                ),
                max_vals,
            )

            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "motion_train_" + suffix_final + ".npy",
                ),
                data_train,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "motion_valid_" + suffix_final + ".npy",
                ),
                data_valid,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "motion_test_" + suffix_final + ".npy",
                ),
                data_test,
            )

            # np.save(os.path.join(os.path.expanduser(TARGET_PATH),
            #                     'context_frames_train_' + suffix_final + '.npy'), context_train)
            # np.save(os.path.join(os.path.expanduser(TARGET_PATH),
            #                     'context_frames_valid_' + suffix_final + '.npy'), context_valid)
            # np.save(os.path.join(os.path.expanduser(TARGET_PATH),
            #                     'context_frames_test_' + suffix_final + '.npy'), context_test)

            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "context_files_train_" + suffix_final + ".npy",
                ),
                files_train,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "context_files_valid_" + suffix_final + ".npy",
                ),
                files_valid,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "context_files_test_" + suffix_final + ".npy",
                ),
                files_test,
            )

            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "clinical_train_" + suffix_final + ".npy",
                ),
                exam_train,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "clinical_valid_" + suffix_final + ".npy",
                ),
                exam_valid,
            )
            np.save(
                os.path.join(
                    os.path.expanduser(TARGET_PATH),
                    "clinical_test_" + suffix_final + ".npy",
                ),
                exam_test,
            )

            if self.display:
                print("Training data has shape: ", data_train.shape)
                print("Validation data has shape: ", data_valid.shape)
                print("Test data has shape: ", data_test.shape)

            self.metadata += "\n\n " + type_name
            self.metadata += "\nTrain shape :" + str(data_train.shape)
            self.metadata += "\nVal shape :" + str(data_valid.shape)
            self.metadata += "\nTest shape :" + str(data_test.shape)

        with open(
            os.path.join(self.exp_name_out, "metadata_" + suffix[:-2] + ".txt"), "a"
        ) as f:
            f.write(self.metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-folder",
        type=str,
        default="data_cleaned_CP\\angles_aligned",
        help="Name of the parrent folder.",
    )
    parser.add_argument(
        "--exp-side", type=str, default="Right", help="Name of the experiment."
    )
    parser.add_argument(
        "--exp-name", type=str, default="cycle", help="Name of the experiment."
    )
    parser.add_argument(
        "--suffix", type=str, default="CP", help="End of the name of the exp."
    )
    parser.add_argument(
        "--num-exp", type=int, default=3, help="number of different splits."
    )
    args = parser.parse_args()

    folder_out = "cycle_CP"

    data_folder = os.path.join(args.exp_folder, args.exp_side)

    spliter = data_spliter(
        data_folder, folder_out, args.suffix, args.exp_name, args.num_exp, display=True
    )
    spliter.split()
