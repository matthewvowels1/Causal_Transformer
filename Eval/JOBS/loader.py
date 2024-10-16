import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_JOBS(random_state=0, version=0):
    # Reorder dimensions so continuous features are first, binary features second
    if version == 0:
        jobs_file_nsw_cont = "nswre74_control.txt"
        jobs_file_nsw_treat = "nswre74_treated.txt"
        reorder_dims = [0, 1, 6, 7, 2, 3, 4, 5]
        bin_feats = [4, 5, 6, 7]
        contfeats = [0, 1, 2, 3]
        colsnws = [
            "t",
            "age",
            "edu",
            "black",
            "hisp",
            "relstat",
            "nodegree",
            "RE74",
            "RE75",
            "RE78",
        ]
    else:
        jobs_file_nsw_cont = "nsw_control.txt"
        jobs_file_nsw_treat = "nsw_treated.txt"
        reorder_dims = [0, 1, 6, 2, 3, 4, 5]
        bin_feats = [3, 4, 5, 6]
        contfeats = [0, 1, 2]
        colsnws = [
            "t",
            "age",
            "edu",
            "black",
            "hisp",
            "relstat",
            "nodegree",
            "RE75",
            "RE78",
        ]

    colsPSID = [
        "t",
        "age",
        "edu",
        "black",
        "hisp",
        "relstat",
        "nodegree",
        "RE74",
        "RE75",
        "RE78",
    ]

    jobs_file_PSID = "psid_controls.txt"

    path = os.path.join(os.path.dirname(__file__), "data")
    # Load data from files
    nsw_treat = pd.read_csv(os.path.join(path, jobs_file_nsw_treat), header=None, sep="  ", engine="python")
    nsw_cont = pd.read_csv(os.path.join(path, jobs_file_nsw_cont), header=None, sep="  ", engine="python")
    psid = pd.read_csv(os.path.join(path, jobs_file_PSID), header=None, sep="  ", engine="python")

    # Assign column names
    nsw_treat.columns = colsnws
    nsw_cont.columns = colsnws
    psid.columns = colsPSID

    # Add 'e' column as designator for control/observational data
    desired_cols = colsnws + ["e"]
    nsw_treat["e"] = 1
    nsw_cont["e"] = 1
    psid["e"] = 0

    # Convert to numpy arrays
    nsw_treat = nsw_treat[desired_cols].values
    nsw_cont = nsw_cont[desired_cols].values
    psid = psid[desired_cols].values

    # Concatenate all datasets
    all_data = np.concatenate((psid, nsw_cont, nsw_treat), axis=0)

    # Separate out the columns
    all_t = all_data[:, 0]        # Treatment indicator
    all_e = all_data[:, -1]       # 'e' column (control/observational data)
    all_y = all_data[:, -2]       # Outcome (RE78 column)

    # Convert 'all_y' to binary, marking positive values as 1, negative as 0
    all_y = (all_y > 0).astype('float64')

    # Extract features (excluding treatment indicator and outcome)
    if version == 0:
        all_x = all_data[:, 1:9]  # Using all columns except 't' and 'e' for features
    else:
        all_x = all_data[:, 1:8]  # Different number of features for version 1

    # Reorder features based on specified order
    all_x = all_x[:, reorder_dims]

    # Split the dataset into training and testing sets
    itr, ite = train_test_split(np.arange(all_x.shape[0]), test_size=0.2, random_state=random_state)

    # Normalize continuous features (first 3 or 4 columns)
    cont_idx = slice(0, 4) if version == 0 else slice(0, 3)
    x_cont_mean = np.mean(all_x[itr, cont_idx], axis=0)
    x_cont_std = np.std(all_x[itr, cont_idx], axis=0)

    # Apply normalization to training and test sets
    all_x[itr, cont_idx] = (all_x[itr, cont_idx] - x_cont_mean) / x_cont_std
    all_x[ite, cont_idx] = (all_x[ite, cont_idx] - x_cont_mean) / x_cont_std

    # Prepare final datasets for training and testing
    xtr, ttr, ytr, etr = all_x[itr], all_t[itr], all_y[itr], all_e[itr]
    xte, tte, yte, ete = all_x[ite], all_t[ite], all_y[ite], all_e[ite]

    # Return data in numpy arrays
    train = (xtr, ttr, ytr, etr)
    test = (xte, tte, yte, ete)

    #print("Data loaded: [X t y e]")

    return train, test, contfeats, bin_feats


if __name__ == "__main__":
    train, test, contfeats, bin_feats = load_JOBS("data")
    for i in range(4):
        print(train[i].shape)
        print(train[i][:10])
    print(contfeats)
    print(bin_feats)
