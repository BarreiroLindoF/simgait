import os

import numpy as np


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_files(name_folder, extension=None):
    def extension_bool(ext):
        if extension == None or ext == extension.lower():
            return True
        else:
            return False

    files = [
        f_name
        for f_name in os.listdir(name_folder)
        if os.path.isfile(os.path.join(name_folder, f_name))
        and extension_bool(f_name.split(".")[-1].lower())
    ]
    return files


def load_csv(file_name, dtype="float", delimiter=",", skiprows=1):
    return np.loadtxt(
        open(file_name, "rb"), dtype=dtype, delimiter=delimiter, skiprows=skiprows
    )


def save_csv(file_name, data, first_row=None, delimiter=", "):
    if file_name[-4:] != ".csv":
        file_name += ".csv"

    skip_enter = True
    with open(file_name, mode="w") as file_:
        if first_row != None:
            skip_enter = False
            file_.write(", ".join(first_row))
        for row in data:
            if not skip_enter:
                file_.write("\n")
            if type(row) is list:
                file_.write(", ".join(row))
            elif type(row) is np.ndarray:
                file_.write(", ".join(list(row)))
            else:
                file_.write(str(row))
            skip_enter = False


def join_path(*args, **kwargs):
    return os.path.join(*args, **kwargs)


def file_write(file_name, content):
    with open(file_name, mode="w") as file_:
        file_.write(content)
