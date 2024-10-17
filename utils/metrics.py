import numpy as np
from numpy._typing import NDArray


def safe_mean(arr):
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return 0
    return np.nanmean(arr)


def risk(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
         y: NDArray[np.float_], t: NDArray[np.int_]) -> float:
    # ypred, y and t should be RCT
    # Adapted from https://github.com/clinicalml/cfrnet/
    # Determine where ypred1 is better than ypred0
    better_pred = ypred1 > ypred0

    # Mean outcome for treated group (t == 1) where ypred1 > ypred0
    y1_mean = safe_mean(y[(t == 1) & better_pred])

    # Mean outcome for control group (t == 0) where ypred1 <= ypred0
    y0_mean = safe_mean(y[(t == 0) & ~better_pred])

    # Proportion of times ypred1 is better than ypred0
    p_fx1 = safe_mean(better_pred)

    # Calculate policy risk (1 - policy value)
    policy_risk = 1 - (y1_mean * p_fx1 + y0_mean * (1 - p_fx1))

    return policy_risk


def eatt(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
         y: NDArray[np.float_], t: NDArray[np.int_]) -> float:
    true_att = safe_mean(y[t == 1]) - safe_mean(y[t == 0])

    estimated_att = safe_mean(ypred1[t == 1] - ypred0[t == 1])

    return abs(true_att - estimated_att)


def eate(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
         ypotential: NDArray[np.float_]) -> dict:
    assert (ypotential.shape[1] == 2)
    true_ate = safe_mean(ypotential[:,1] - ypotential[:,0])

    estimated_ate = safe_mean(ypred1 - ypred0)

    return abs(true_ate - estimated_ate)


def pehe(ypred1: NDArray[np.float_], ypred0: NDArray[np.float_],
         ypotential: NDArray[np.float_]) -> dict:
    assert (ypotential.shape[1] == 2)
    true_cate = ypotential[:, 1] - ypotential[:, 0]
    est_cate = ypred1 - ypred0

    return np.sqrt(np.mean((true_cate - est_cate) ** 2))
