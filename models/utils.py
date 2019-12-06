import numpy as np

def load_csv(file_name, dtype='float', delimiter=",", skiprows=1):
    return np.loadtxt(open(file_name, "rb"), dtype=dtype, delimiter=delimiter, skiprows=skiprows)

def save_csv(file_name, data, first_row=None, delimiter=", "):
    if file_name[-4:] != '.csv':
        file_name += '.csv'

    skip_enter = True
    with open(file_name, mode='w') as file_:
        if first_row != None:
            skip_first_enter = False
            file_.write(", ".join(first_row))
        for row in data:
            if not skip_enter:
                file_.write('\n')
            if type(row) is list:
                file_.write(", ".join(row))
            elif type(row) is np.ndarray:
                file_.write(", ".join(list(row)))
            else:
                file_.write(str(row))
            skip_enter = False
