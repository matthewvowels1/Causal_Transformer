import os
import pandas as pd


def load_ACIC(experiment=0, replication=0):
    assert experiment in range(77)
    assert replication in range(10)

    # Load confounders
    path = os.path.join(os.path.dirname(__file__), f"data/confounders.csv")
    X = pd.read_csv(path, engine="python")
    non_numeric_cols = X.select_dtypes(include=[object]).columns
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

    # Load t y
    path = os.path.join(os.path.dirname(__file__), f"data/simulation_param_{experiment + 1}_simu_{replication + 1}.csv")
    Y = pd.read_csv(path, engine="python")
    t_y = Y[['z', 'y']].rename(columns={'z': 't'})
    dataset = pd.concat([X, t_y], axis=1).to_numpy(dtype='float32')

    potential_y = Y[['mu.0', 'mu.1']].to_numpy(dtype='float32')

    var_types = {var: '?' for var in X.columns}
    var_types['t'] = 'bin'
    var_types['y'] = 'cont'

    return dataset, potential_y, var_types


if __name__ == "__main__":
    dataset, potential_y, var_types = load_ACIC()
    print(dataset.dtype)
