import numpy as np

markers_names = ["LTOE", "LHEE", "LANK", "LKNE", "LASI", "LFIN", "LWRA", "LWRB",
                 "LELB", "LSHO", "LFHD", "LBHD", "T10", "STRN", "CLAV", "C7", 
                 "RTOE", "RHEE", "RANK", "RKNE", "RASI", "RFIN", "RWRA", "RWRB",
                 "RELB", "RSHO", "RFHD", "RBHD"]

def mid_point(*argv):
    points = np.array(argv)
    return points.sum(0) / points.shape[0]


def reduce_data(data, markers):
    new_data = []
    for frame in range(data.shape[0]):
        new_marker_data = []
        for marker in markers:
            if "_" in marker:
                markers_list = marker.split("_")
                markers_list = [
                    data[frame, markers_names.index(mark), :] for mark in markers_list
                ]

                markers_list = np.stack(markers_list)
                new_marker = markers_list.sum(0) / markers_list.shape[0]
                new_marker_data.append(new_marker)
            else:
                new_marker_data.append(data[frame, markers_names.index(marker), :])
        new_data.append(np.stack(new_marker_data))
    new_data = np.stack(new_data)
    return new_data


if __name__ == "__main__":
    path = "/Users/joao/workspace/bitbucket/joao/code/gait_analysis/data_extracted_CP_2/markers/00021_00081_20071128-GBNNN-VDEF-07.npy"

    data = np.load(path).swapaxes(0, 1)
    data = reduce_data(data)
    print(data.shape)

    max_z = data[:, :, 2].max()

    data /= max_z

    # print(max_z)

    # from motion3d import Motion

    # motion = Motion(data, markers_names_final, "adjacency.json")
    # motion.render_frame_3D(4)


    print()

    import os
    path = "/Users/joao/workspace/bitbucket/joao/code/gait_analysis/data_extracted_CP_2/markers"
    data_list = []
    for f_name in os.listdir(path):
        if f_name[-3:] == "npy":
            data = np.load(os.path.join(path, f_name))
            if len(data.shape):
                data = data.swapaxes(0, 1)
                data = reduce_data(data)
                max_z = data[:, :, 2].max()
                data /= max_z
                data_list.append(data)
            else:
                print("bad", f_name)
    data = np.concatenate(data_list)
    print(data.shape)
