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


def compute(seeds, model_constructor):
    results = [evaluate(model_constructor=model_constructor,
                        seed=seed) for seed in seeds]
    proper_dict = {key: [d[key] for d in results] for key in results[0]}
    return proper_dict


for model in models:
    proper_dict = compute(seeds=seeds, model_constructor=models_methods[model])
    with open(output_path, "a") as file:
        file.write(f"\nmodel: {model}\nseeds: {seeds}\n")
        for name, values in proper_dict.items():
            file.write(f"{name}: {np.mean(values):.4f} +- {np.std(values):.4f}  n={len(values)}\n")
