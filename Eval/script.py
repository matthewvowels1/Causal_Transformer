import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Eval.ACIC16.eval import evaluate as acic_eval
from Eval.IHDP.eval import evaluate as ihdp_eval
from Eval.twins.eval import evaluate as twins_eval
from Eval.JOBS.eval import evaluate as jobs_eval
from Eval.eval_utils import instantiate_old_CFCN, instantiate_new_CFCN, instantiate_CaT
from baseline.transformer import TransformerRegressor


# take care because y is in the input for Cat and CFCN but not for RF and transformer
def instantiate_rf(DAGnx, var_types):
    return RandomForestRegressor(random_state=0)


def instantiate_transformer(DAGnx, var_types):
    return TransformerRegressor(num_variables=len(var_types) - 1, input_dim=1)


def generator_CaT_dropout(dropout_rate):
    return lambda DAGnx, var_types: instantiate_CaT(DAGnx=DAGnx, var_types=var_types, dropout_rate=dropout_rate)


available_evaluations = {'ihdp': ihdp_eval, 'twins': twins_eval, 'jobs': jobs_eval, 'acic': acic_eval}

available_models = {'CFCN': instantiate_old_CFCN,
                    'CFCN mod': instantiate_new_CFCN,
                    'CaT': instantiate_CaT,
                    'CaT dropout 0.2': generator_CaT_dropout(0.2),
                    'CaT dropout 0.1': generator_CaT_dropout(0.1),
                    'random forest': instantiate_rf,
                    'transformer': instantiate_transformer}

evaluations = ['acic']
models = ['random forest', 'CFCN', 'CFCN mod', 'CaT']

output_path = 'results.txt'

if os.path.exists(output_path):
    raise FileExistsError(output_path)

for model in models:
    for evaluation in evaluations:
        print(f"{evaluation} {model}")
        results = available_evaluations[evaluation](model_constructor=available_models[model])
        with open(output_path, "a") as file:
            file.write(f"\nevaluation {evaluation}\nmodel: {model}\n")
            for name, values in results.items():
                file.write(f"{name}: {np.mean(values):.4f} +- {np.std(values):.4f}  n={len(values)}\n")
