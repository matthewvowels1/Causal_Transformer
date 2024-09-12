import numpy as np
import os


def load_IHDP(path="IHDP/", replication=1, split='train'):
    path_data = os.path.join(path, f"ihdp_npci_{split}_{replication}.csv")
    data = np.loadtxt(path_data, delimiter=',', skiprows=1)

    x, y = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
    y0, y1, z = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    # custom fix
    z[:, 13] -= 1

    return z, x, y, y1, y0
