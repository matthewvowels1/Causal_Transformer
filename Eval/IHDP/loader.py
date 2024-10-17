import numpy as np
import os


def load_IHDP(replication=1, split='train'):
    path = os.path.join(os.path.dirname(__file__), f"data/ihdp_npci_{split}_{replication}.csv")
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    t, y = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
    y0, y1, z = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    # custom fix
    z[:, 13] -= 1

    return z, t, y, y1, y0

