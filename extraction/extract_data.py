# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
from numpy import savetxt

try:
    from my_utils import is_float, join_path, save_csv
    from btk import btk
except:
    raise

FILES_TO_REMOVE = [
    "00099_00634_20051125-GBNNN-VDNN-15.C3D",
    "00099_00634_20051125-GBNNN-VDNN-16.C3D",
    "00099_00634_20051125-GBNNN-VDNN-17.C3D",
    "00099_00634_20051125-GBNNN-VDNN-18.C3D",
    "00099_00634_20051125-GBNNN-VDNN-20.C3D",
    "00099_00634_20051125-GBNNN-VDNN-21.C3D",
    "00761_01211_19980219-GBNNN-NMNN-02.C3D",
    "00120_00397_20051102-GBNNN-VDNN-09.C3D",
    "00041_02782_20121015-GBASN-VDEF-05.C3D",
    "00250_00353_20080714-GBNNT-VDEF-04.C3D",
    "00070_00303_20080326-GBNNN-VDNN-02.C3D",
]


class Extracter:
    def __init__(self, datasets, keep_pathology=["CP_Spastic_Uni_Hemiplegia"]):
        self.datasets = datasets
        self.keep_pathology = keep_pathology

        self.btk_reader = btk.btkAcquisitionFileReader()
        self.metadata_reader = None

        self.n_total_files = 0
        self.files = self._get_files()
        self.data = None
        self.list_labels = None
        self.angles_names = None
        self.markers_names = None

        # print(
        #     "We have found {} files and keep {} !".format(
        #         self.n_total_files, len(self.files)
        #     )
        # )

    def _keep_file(self, path, dataset, file_name):
        # check if it's a file
        is_file = os.path.isfile(join_path(path, file_name))
        # check if it's a c3d
        is_c3d = file_name[-3:].lower() == "c3d"
        # check if the pathology is what we need
        if dataset == "Healthy":
            keep_patho = True
        else:
            keep_patho = (
                    self.get_diagnostic(join_path(path, file_name)) in self.keep_pathology
            )

        # check if file isn't in remove files
        is_to_remove = file_name not in FILES_TO_REMOVE
        # check end
        if dataset == "CP_Gait_1.0":
            end_file = file_name.split("-")[-2]
            if end_file != "VDEF":
                return False
        return is_file and is_c3d and keep_patho and is_to_remove

    def _get_files(self):
        """."""
        files = []
        for path in self.datasets:
            for f_name in os.listdir(path):
                self.n_total_files += 1
                # print("SEARCHING files ...", end="\r")
                dataset = path.split("/")[-1]
                if self._keep_file(path, dataset, f_name):
                    files.append(join_path(path, f_name))
        # print("".ljust(70), end="\r")
        return files

    def _update_btk(self, file_path):
        """."""
        self.btk_reader.SetFilename(file_path)
        self.btk_reader.Update()
        acq = self.btk_reader.GetOutput()
        self.metadata_reader = acq.GetMetaData()

    def extract(
            self,
            examination=False,
            diagnostic=False,
            affected_side=False,
            gmfcs=False,
            markers=False,
            angles=False,
            events=False,
            all_=False,
    ):
        self.data = self.init_data(locals())

        if examination or all_:
            self.list_labels = self.get_labels_examination()
        if angles or all_:
            if self.angles_names is None:
                events = True
                self.angles_names = self.get_labels_angles()

        for file_id in range(len(self.files)):
            dataset = self.files[file_id].split("\\")[-2]
            # print("EXTRACTING FILE {} OVER {}".format(file_id, len(self.files)), end='\r')

            self.add_file(self.files[file_id])
            if examination or all_:
                self.data["examination"][-1] = self.get_examination(self.files[file_id])
            if diagnostic or all_:
                if dataset == "Healthy":
                    self.data["diagnostic"][-1] = dataset
                else:
                    self.data["diagnostic"][-1] = self.get_diagnostic(
                        self.files[file_id]
                    )
            if affected_side or all_:
                if dataset == "Healthy":
                    self.data["affected_side"][-1] = "normal"
                else:
                    self.data["affected_side"][-1] = self.get_affected_side(
                        self.files[file_id]
                    )
            if gmfcs or all_:
                self.data["gmfcs"][-1] = self.get_gmfcs(self.files[file_id])
            if markers or all_:
                data = self.get_markers(self.files[file_id])
                if data is not None:
                    self.data["markers"][-1] = self.get_markers(self.files[file_id])
            '''
            if angles or all_:
                self.data["angles"][-1] = self.get_angles(self.files[file_id])
            '''
            if events or all_:
                self.data["events"][-1] = self.get_events(self.files[file_id])
        # print("".ljust(70), end="\r")

    def get_examination(self, file_path):
        """Examination data starts with A_ or C_."""
        self._update_btk(file_path)

        features = ["NaN"] * len(self.list_labels)

        meta_subject = self.metadata_reader.FindChild("SUBJECTS").value()
        n_childs = meta_subject.GetChildNumber()

        for child_id in range(1, n_childs):
            label = meta_subject.GetChild(child_id).GetLabel()
            if label.lower() in self.list_labels:
                value = (
                    meta_subject.FindChild(label)
                        .value()
                        .GetInfo()
                        .ToString()[0]
                        .replace(" ", "")
                )

                if value[-1] in "-+_*°" and len(value) >= 1:
                    symb = value[-1]
                    value = value[:-1]
                    if is_float(value):
                        value = float(value)
                        if symb == "+":
                            value = value + 0.25
                        elif symb == "-":
                            value = value - 0.25
                        value = str(value)
                if value.lower() == "nt":
                    value = "NaN"
                if len(value) == 0 or value[-1] in "-+_*°" and len(value) == 1:
                    value = "NaN"
                if value == "null":
                    value = "NaN"
                label_idx = self.list_labels.index(label.lower())
                features[label_idx] = value

        return features

    def get_diagnostic(self, file_path):
        self._update_btk(file_path)
        diag = (
            self.metadata_reader.FindChild("SUBJECTS")
                .value()
                .FindChild("DIAGNOSTIC")
                .value()
                .GetInfo()
                .ToString()[0]
        )
        return diag

    def get_affected_side(self, file_path):
        self._update_btk(file_path)
        aff_side = (
            self.metadata_reader.FindChild("SUBJECTS")
                .value()
                .FindChild("AFFECTED_SIDE")
                .value()
                .GetInfo()
                .ToString()[0]
        )
        aff_side = aff_side.replace("\r", " ")
        if aff_side == "Null":
            aff_side = file_path.split("/")[-2]
            aff_side = "_".join(aff_side.split("_")[:-2])
        return aff_side

    def get_gmfcs(self, file_path):
        self._update_btk(file_path)
        gmfcs = (
            self.metadata_reader.FindChild("SUBJECTS")
                .value()
                .FindChild("GMFCS")
                .value()
                .GetInfo()
                .ToString()[0]
        )
        return gmfcs

    def get_labels_examination(self):
        labels_list = []

        for file_id in range(len(self.files)):
            # print(
            #     "SEARCHING FEATURES FOR EXAMINATION IN FILE {} OVER {}".format(
            #         file_id, len(self.files)
            #     ),
            #     end="\r",
            # )

            labels_list_tmp = []
            reader = btk.btkAcquisitionFileReader()
            reader.SetFilename(self.files[file_id])
            reader.Update()
            acq = reader.GetOutput()
            metadata = acq.GetMetaData()

            meta_subject = metadata.FindChild("SUBJECTS").value()
            n_childs = meta_subject.GetChildNumber()

            for child_id in range(1, n_childs):
                label = meta_subject.GetChild(child_id).GetLabel()

                if label[:2] == "A_" or label[:2] == "C_":
                    labels_list_tmp.append(label.lower())

            labels_list = list(set(labels_list + labels_list_tmp))

        labels_list.sort()
        labels_list = ["age", "sex"] + labels_list

        # print("".ljust(70), end="\r")
        return labels_list

    def get_labels_angles(self):
        """."""
        labels_list = []
        files_ = self.files[:]
        for file_id in range(len(self.files)):
            # print(
            #     "SEARCHING NAMES FOR ANGLES IN FILE {} OVER {}".format(
            #         file_id, len(self.files)
            #     ),
            #     end="\r",
            # )
            reader = btk.btkAcquisitionFileReader()
            reader.SetFilename(self.files[file_id])
            reader.Update()
            acq = reader.GetOutput()
            metadata = acq.GetMetaData()
            labels_list_tmp = list(
                metadata.FindChild("POINT")
                    .value()
                    .FindChild("ANGLES")
                    .value()
                    .GetInfo()
                    .ToString()
            )
            labels_list_tmp = [name.replace(" ", "") for name in labels_list_tmp]
            if len(labels_list_tmp) > 16:
                if not len(labels_list):
                    labels_list = labels_list_tmp[:]
                else:
                    labels_list = set(labels_list).intersection(labels_list_tmp)
            else:
                files_.remove(self.files[file_id])
            # if not len(labels_list):  labels_list = labels_list_tmp[:]
            # else: labels_list = set(labels_list).intersection(labels_list_tmp)
        # print("".ljust(70), end="\r")

        self.files = files_[:]
        return list(labels_list)

    def get_markers(self, file_path):
        self._update_btk(file_path)
        try:
            markers_data = [self.btk_reader.GetOutput().GetPoint(marker).GetValues() for marker in self.markers_names]
        except:
            return None
        markers_data = np.array(markers_data)
        return markers_data

    def get_angles(self, file_path):
        self._update_btk(file_path)
        # get data for each marker
        angles_data = [
            self.btk_reader.GetOutput().GetPoint(angle).GetValues()
            for angle in self.angles_names
        ]
        angles_data = np.array(angles_data)
        return angles_data

    def get_events(self, file_path):
        acq = self.btk_reader.GetOutput()
        n_events = acq.GetEventNumber()
        event_frames = [acq.GetEvent(event).GetFrame() for event in range(n_events)]
        event_side = list(
            self.metadata_reader.FindChild("EVENT")
                .value()
                .FindChild("CONTEXTS")
                .value()
                .GetInfo()
                .ToString()
        )
        event_side = [name.replace(" ", "") for name in event_side]
        event_type = list(
            self.metadata_reader.FindChild("EVENT")
                .value()
                .FindChild("LABELS")
                .value()
                .GetInfo()
                .ToString()
        )
        event_type = [name.replace(" ", "") for name in event_type]
        event_frames, event_side, event_type = zip(
            *sorted(zip(event_frames, event_side, event_type))
        )

        events_data = []
        events_data.append(
            ["frames"] + (np.array(event_frames) - acq.GetFirstFrame()).tolist()
        )
        events_data.append(["side"] + list(event_side))
        events_data.append(["type"] + list(event_type))

        events_data = np.array(events_data).T.tolist()
        return events_data

    def init_data(self, variables):
        data = {}
        data["files"] = []
        keys = [key for key, value in variables.items() if value == True]
        if "all_" in keys:
            keys = [key for key in variables if key != "self" and key != "all_"]
        if "angles" in keys:
            keys.append("events")
        for key in keys:
            data[key] = []

        return data

    def add_file(self, file_path):
        for key in self.data.keys():
            if key == "files":
                self.data[key].append(file_path.split("/")[-1])
            else:
                self.data[key].append(None)

    def save_csv(folder_tmp, files_name, data, first_row):
        # savetxt(path, data)
        return None

    def save(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for key in self.data.keys():
            path = os.path.join(output_folder, key)
            if key == "angles" or key == "markers":
                folder_tmp = join_path(output_folder, key)
                if not os.path.exists(folder_tmp):
                    os.makedirs(folder_tmp)
                for idx in range(len(self.data[key])):
                    np.save(
                        join_path(folder_tmp, self.data["files"][idx].split("\\")[-1].split('.')[0]),
                        self.data[key][idx],
                    )

            elif key == "events":
                folder_tmp = join_path(output_folder, "events")
                if not os.path.exists(folder_tmp):
                    os.makedirs(folder_tmp)
                for idx in range(len(self.data["events"])):
                    save_csv(
                        join_path(folder_tmp, self.data["files"][idx].split("\\")[-1].split('.')[0]),
                        self.data["events"][idx][1:],
                        first_row=self.data["events"][idx][0],
                    )
            else:
                with open("{}.csv".format(path), mode="w") as file_:
                    if key == "examination":
                        file_.write(", ".join(self.list_labels))
                    else:
                        file_.write(key.upper())
                    for row in self.data[key]:
                        file_.write("\n")
                        if type(row) is list:
                            file_.write(", ".join(row))
                        else:
                            file_.write(row)


if __name__ == "__main__":
    currPath = os.path.dirname(os.getcwd())
    datasets = [os.path.expanduser(currPath + "\\data\\raw\\CP\\CP_Gait_1.0")]
    keep_pathology = ["CP_Spastic_Uni_Hemiplegia", "CP_Spastic_Diplegia"]

    extracter = Extracter(datasets, keep_pathology=keep_pathology)
    extracter.angles_names = [
        "RAnkleAngles",
        "LAnkleAngles",
        "RHipAngles",
        "LHipAngles",
        "RPelvisAngles",
        "LKneeAngles",
        "RKneeAngles",
        "LPelvisAngles",
    ]
    extracter.markers_names = [
        "LTOE",
        "LHEE",
        "LANK",
        "LKNE",
        "LASI",
        "LFIN",
        "LWRA",
        "LWRB",
        "LELB",
        "LSHO",
        "LFHD",
        "LBHD",
        "T10",
        "STRN",
        "CLAV",
        "C7",
        "RTOE",
        "RHEE",
        "RANK",
        "RKNE",
        "RASI",
        "RFIN",
        "RWRA",
        "RWRB",
        "RELB",
        "RSHO",
        "RFHD",
        "RBHD",
    ]

    extracter.extract(all_=True)
    # extracter.extract(examination=False, diagnostic=True, affected_side=False, gmfcs=False, angles=True, events=False, all_=False)
    # print("EXTRACTER DATA", extracter.data["affected_side"])
    extracter.save(output_folder=currPath + "\data\extracted\CP")
