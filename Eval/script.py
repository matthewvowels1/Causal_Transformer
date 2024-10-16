import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Eval.IHDP.eval import evaluate as ihdp_eval
from Eval.twins.eval import evaluate as twins_eval
from Eval.JOBS.eval import evaluate as jobs_eval
from Eval.eval_utils import instantiate_old_CFCN, instantiate_new_CFCN, instantiate_CaT
from baseline.transformer import TransformerRegressor

# take care because y is in the input for Cat and CFCN but not for RF and transformer
def instantiate_rf(DAGnx,var_types):
    return RandomForestRegressor(random_state=0)
def instantiate_transformer(DAGnx,var_types):
    return TransformerRegressor(input_dim=len(var_types)-1)

available_evaluations = {'ihdp': ihdp_eval, 'twins': twins_eval, 'jobs': jobs_eval}

available_models = {'CFCN': instantiate_old_CFCN,
                  'CFCN mod': instantiate_new_CFCN,
                  'CaT': instantiate_CaT,
                  'random forest': instantiate_rf,
                  'transformer': instantiate_transformer}

evaluations = ['ihdp','twins','jobs']
models = ['random forest', 'transformer']

output_path = 'results.txt'

if os.path.exists(output_path):
    pass
    #raise FileExistsError(output_path)

warnings.simplefilter("once",UserWarning)

for evaluation in evaluations:
    for model in models:
        results = available_evaluations[evaluation](model_constructor=available_models[model])
        with open(output_path, "a") as file:
            file.write(f"\nevaluation {evaluation}\nmodel: {model}\n")
            for name, values in results.items():
                file.write(f"{name}: {np.mean(values):.4f} +- {np.std(values):.4f}  n={len(values)}\n")
