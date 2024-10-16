"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

Note: Load real-world individualized treatment effects estimation datasets

(1) data_loading_twin: Load twins data.
  - Reference: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html
"""
import os

# Necessary packages
import numpy as np
from scipy.special import expit


def load_twins():
    """Load twins data.

    Args:
      - train_rate: the ratio of training data

    Returns:
      - train_x: features in training data
      - train_t: treatments in training data
      - train_y: observed outcomes in training data
      - train_potential_y: potential outcomes in training data
      - test_x: features in testing data
      - test_potential_y: potential outcomes in testing data
    """
    path = os.path.join(os.path.dirname(__file__), f"data/twins.csv")
    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    ori_data = np.loadtxt(path, delimiter=",", skiprows=1)

    # Define features
    x = ori_data[:, :30]
    no, dim = x.shape

    # Define potential outcomes
    potential_y = ori_data[:, 30:]
    # Die within 1 year = 1, otherwise = 0
    potential_y = np.array(potential_y < 9999, dtype=float)

    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
    prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

    prob_t = prob_temp / (2 * np.mean(prob_temp))
    prob_t[prob_t > 1] = 1

    t = np.random.binomial(1, prob_t, [no, 1])
    t = t.reshape([no, ])

    ## Define observable outcomes
    y = np.zeros([no, 1])
    y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
    y = np.reshape(np.transpose(y), [no, ])

    dataset = np.concatenate((x,t[:,None], y[:, None]), axis=1)

    var_types = {
        'dtotord': 'ord',
        'anemia': 'bin',
        'cardiac': 'bin',
        'lung': 'bin',
        'diabetes': 'bin',
        'herpes': 'bin',
        'hydra': 'bin',
        'hemo': 'bin',
        'chyper': 'bin',
        'phyper': 'bin',
        'eclamp': 'bin',
        'incervix': 'bin',
        'pre4000': 'bin',
        'preterm': 'bin',
        'renal': 'bin',
        'rh': 'bin',
        'uterine': 'bin',
        'othermr': 'bin',
        'cigar': 'cat',
        'drink': 'cat',
        'wtgain': 'num',
        'pldel': 'cat',
        'gestat': 'cat',
        'dmage': 'num',
        'dmeduc': 'cat',
        'dmar': 'bin',
        'resstatb': 'cat',
        'mpcb': 'cat',
        'nprevist': 'ord',
        'adequacy': 'cat',
        't': 'bin',
        'y': 'bin',
    }

    return dataset, potential_y, var_types
