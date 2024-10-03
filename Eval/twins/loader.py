"""
File for loading the Twins semi-synthetic (treatment is simulated) dataset.

Louizos et al. (2017) introduced the Twins dataset as an augmentation of the
real data on twin births and twin mortality rates in the USA from 1989-1991
(Almond et al., 2005). The treatment is "born the heavier twin" so, in one
sense, we can observe both potential outcomes. Louizos et al. (2017) create an
observational dataset out of this by hiding one of the twins (for each pair) in
the dataset. Furthermore, to make sure the twins are very similar, they limit
the data to the twins that are the same sex. To look at data with higher
mortality rates, they further limit the dataset to twins that were born weighing
less than 2 kg. To ensure there is some confounding, Louizos et al. (2017)
simulate the treatment assignment (which twin is heavier) as a function of the
GESTAT10 covariate, which is the number of gestation weeks prior to birth.
GESTAT10 is highly correlated with the outcome and it seems intuitive that it
would be a cause of the outcome, so this should simulate some confounding.

References:

    Almond, D., Chay, K. Y., & Lee, D. S. (2005). The costs of low birth weight.
        The Quarterly Journal of Economics, 120(3), 1031-1083.

    Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M.
        (2017). Causal effect inference with deep latent-variable models. In
        Advances in Neural Information Processing Systems (pp. 6446-6456).
"""

import pandas as pd


def load_twins(datapath="data/twins.csv", data_format='numpy',
               return_sketchy_ites=False, return_sketchy_ate=False):
    """
    Load the Twins dataset

    :param datapath: path to folder for data
    :param return_sketchy_ites: if True, return sketchy ITEs
    :param return_sketchy_ate: if True, return sketchy ATE
    :return: dictionary of results
    """

    assert data_format in ('numpy', 'pandas'), f"unknown data format {data_format}, should be numpy or pandas"

    full_df = pd.read_csv(datapath, index_col=0)

    new_df = full_df.drop(['y0', 'y1', 'y_cf', 'Propensity'], axis='columns').rename(columns={'T': 't', 'yf': 'y'})

    if return_sketchy_ites or return_sketchy_ate:
        ites = full_df['y1'] - full_df['y0']
        ites_np = ites.to_numpy()
        if return_sketchy_ites:
            new_df['ites'] = ites
        if return_sketchy_ate:
            new_df['ate'] = ites_np.mean()

    if data_format == 'numpy':
        new_df.to_numpy()

    var_types = {
        'eclamp': 'bin',
        'gestatcat1': 'cat',
        'gestatcat2': 'cat',
        'gestatcat3': 'cat',
        'gestatcat4': 'cat',
        'gestatcat5': 'cat',
        'gestatcat6': 'cat',
        'gestatcat7': 'cat',
        'gestatcat8': 'cat',
        'gestatcat9': 'cat',
        'gestatcat10': 'cat',
        'gestatcat1.1': 'cat',
        'gestatcat2.1': 'cat',
        'gestatcat3.1': 'cat',
        'gestatcat4.1': 'cat',
        'gestatcat5.1': 'cat',
        'gestatcat6.1': 'cat',
        'gestatcat7.1': 'cat',
        'gestatcat8.1': 'cat',
        'gestatcat9.1': 'cat',
        'gestatcat10.1': 'cat',
        'gestatcat1.2': 'cat',
        'gestatcat2.2': 'cat',
        'gestatcat3.2': 'cat',
        'gestatcat4.2': 'cat',
        'gestatcat5.2': 'cat',
        'gestatcat6.2': 'cat',
        'gestatcat7.2': 'cat',
        'gestatcat8.2': 'cat',
        'gestatcat9.2': 'cat',
        'gestatcat10.2': 'cat',
        'bord': 'bin',
        'othermr': 'bin',
        'dmar': 'bin',
        'csex': 'bin',
        'cardiac': 'bin',
        'uterine': 'bin',
        'lung': 'bin',
        'diabetes': 'bin',
        'herpes': 'bin',
        'anemia': 'bin',
        'hydra': 'bin',
        'chyper': 'bin',
        'phyper': 'bin',
        'incervix': 'bin',
        'pre4000': 'bin',
        'preterm': 'bin',
        'renal': 'bin',
        'rh': 'bin',
        'hemo': 'bin',
        'tobacco': 'bin',
        'alcohol': 'bin',
        'orfath': 'cat',
        'adequacy': 'cat',
        'drink5': 'cat',
        'mpre5': 'cat',
        'meduc6': 'cat',
        'mrace': 'cat',
        'ormoth': 'cat',
        'frace': 'cat',
        'birattnd': 'cat',
        'stoccfipb_reg': 'cat',
        'mplbir_reg': 'cat',
        'cigar6': 'cat',
        'mager8': 'cat',
        'pldel': 'cat',
        'brstate_reg': 'cat',
        'feduc6': 'cat',
        'dfageq': 'cat',
        'nprevistq': 'cat',
        'data_year': 'cat',
        'crace': 'cat',
        'birmon': 'cyc',
        'dtotord_min': 'ord',
        'dlivord_min': 'ord',
        't': 'bin',
        'y': 'bin'
    }
    return new_df, var_types


if __name__ == "__main__":
    df, var_types = load_twins(data_format='pandas', return_sketchy_ate=True, return_sketchy_ites=True)
    print(df.columns)

    print(df[['bord','y','t']][:10])
