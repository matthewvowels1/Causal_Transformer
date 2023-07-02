import numpy as np
import os

def sigm(x):
    return 1/(1 + np.exp(-x))

def inv_sigm(x):
    return np.log(x/(1-x))


def generate_data(N, seed, dataset):
    np.random.seed(seed=seed)
    if dataset == 'general':
        # confounders
        z1 = np.random.binomial(1, 0.5, (N, 1))
        z2 = np.random.binomial(1, 0.65, (N, 1))
        z3 = np.round(np.random.uniform(0, 4, (N, 1)), 0)
        z4 = np.round(np.random.uniform(0, 5, (N, 1)), 0)
        uz5 = np.random.randn(N, 1)
        z5 = 0.2 * z1 + uz5
        Z = np.concatenate([z1, z2, z3, z4, z5], 1)

        # risk vars:
        r1 = np.random.randn(N, 1)
        r2 = np.random.randn(N, 1)
        R = np.concatenate([r1, r2], 1)

        # instrumental vars:
        i1 = np.random.randn(N, 1)
        i2 = np.random.randn(N, 1)
        I = np.concatenate([i1, i2], 1)

        # treatment:
        ux = np.random.randn(N, 1)
        xp = sigm(-5 + 0.05 * z2 + 0.25 * z3 + 0.6 * z4 + 0.4 * z2 * z4 + 0.15 * z5 + 0.1 * i1 + 0.15 * i2 + 0.1 * ux)
        X = np.random.binomial(1, xp, (N, 1))

        # mediator:
        Um = np.random.randn(N, 1)
        m1 = 0.8 + 0.15 * Um
        m0 = 0.15 * Um
        M = m1 * X + m0 * (1 - X)

        # outcomes:
        Y1 = np.random.binomial(1, sigm(np.exp(-1 + m1 - 0.1 * z1 + 0.35 * z2 +
                                               0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2)),
                                (N, 1))
        Y0 = np.random.binomial(1,
                                sigm(-1 + m0 - 0.1 * z1 + 0.35 * z2 + 0.25 * z3 + 0.2 * z4 + 0.15 * z2 * z4 + r1 + r2),
                                (N, 1))
        Y = Y1 * X + Y0 * (1 - X)

        # colliders:
        C = 0.6 * Y + 0.4 * X + 0.4 * np.random.randn(N, 1)


        # DAG creation (there is a one in position (i,j) if there is an arrow from i to j and zero otherwise)
        num_vars = Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1] + C.shape[1]
        DAG = np.zeros((num_vars, num_vars))

        # set Z -> X and Z -> Y links
        DAG[0:Z.shape[1], Z.shape[1] + R.shape[1] + I.shape[1] : Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]] = 1
        DAG[0:Z.shape[1], Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] : Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1]] = 1

        # set R -> Y links
        DAG[Z.shape[1]:(Z.shape[1] + R.shape[1]),
        Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] : Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1]] = 1

        # set I -> X links
        DAG[(Z.shape[1] + R.shape[1]):(Z.shape[1] + R.shape[1] + I.shape[1]),
        Z.shape[1] + R.shape[1] + I.shape[1] : Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]] = 1

        # set X -> M links
        DAG[(Z.shape[1] + R.shape[1] + I.shape[1]):(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]),
        (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]) : (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1])] = 1

        # set X->C links
        DAG[(Z.shape[1] + R.shape[1] + I.shape[1]):(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]),
        (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1] ):(
                    Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1] + C.shape[1])] = 1

        # set X->Y links
        DAG[(Z.shape[1] + R.shape[1] + I.shape[1]):(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]),
        (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1]):(
                    Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1])] = 1

        # set M -> Y links
        DAG[(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1]):(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1]),
        (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1]):(
		        Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1])] = 1

        # set Y -> C links
        DAG[(Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1]):(
			        Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1]),
        (Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1]):(
		        Z.shape[1] + R.shape[1] + I.shape[1] + X.shape[1] + M.shape[1] + Y.shape[1] + C.shape[1])] = 1

    return np.concatenate([Z, R, I, X, M, Y, C, Y1, Y0], 1), DAG