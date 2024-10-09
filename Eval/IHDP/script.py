from Eval.IHDP.eval import evaluate
from Eval.eval_utils import instantiate_old_CFCN, instantiate_new_CFCN, compute_result, instantiate_CaT
import itertools

models = {'old CFCN': instantiate_old_CFCN,
          'new CFCN': instantiate_new_CFCN,
          'CaT': instantiate_CaT}

parameters = {'model': ['old CFCN', 'new CFCN', 'CaT'],
              'seed': [0],
              'dropout_rate': [0]}

param_keys = parameters.keys()
param_values = parameters.values()

# Generate all combinations of the parameter values
combinations = itertools.product(*param_values)

for combination in combinations:
    param_dict = dict(zip(param_keys, combination))
    print(param_dict)
    evaluate(model_constructor=lambda **kwargs: models[param_dict['model']](**kwargs,
                                                                            dropout_rate=param_dict['dropout_rate']),
             seed=param_dict['seed'])
    compute_result(output_path=param_dict.__str__())
