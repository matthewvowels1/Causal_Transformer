import os

import numpy as np

from Eval.twins.eval import evaluate
from Eval.eval_utils import instantiate_old_CFCN, instantiate_new_CFCN, instantiate_CaT

models_methods = {'old CFCN': instantiate_old_CFCN,
                  'new CFCN': instantiate_new_CFCN,
                  'CaT': instantiate_CaT}

models = ['old CFCN', 'new CFCN', 'CaT']
seeds = range(10)

output_path = 'results/results.txt'

if os.path.exists(output_path):
    raise FileExistsError(output_path)



for model in models:
    proper_dict = evaluate(seeds=seeds, model_constructor=models_methods[model])
    with open(output_path, "a") as file:
        file.write(f"\nmodel: {model}\nseeds: {seeds}\n")
        for name, values in proper_dict.items():
            file.write(f"{name}: {np.mean(values):.4f} +- {np.std(values):.4f}  n={len(values)}\n")
