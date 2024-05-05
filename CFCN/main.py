import argparse
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from trainer import objective


def parse_layers(value):
    # This function takes a string and converts it to a list of integers
    try:
        # Split the string by comma and convert each item to an integer
        layers = list(map(int, value.split(',')))
        if not all(n > 0 for n in layers):
            raise ValueError("All numbers of neurons must be positive.")
        return layers
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid layer configuration: {value}. Error: {e}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='simple_test_v2',
        required=True,
        help="Name of the dataset to be used for training and/or testing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to train and/or run the model ('cuda' or 'cpu')."
    )

    parser.add_argument(
        "--existing_model_path",
        type=str,
        required=False,
        default='None',
        help="Path to a model checkpoint file (if not specified, will use random init).",
    )

    parser.add_argument(
        "--model_save_path",
        type=Path,
        required=False,
        default='checkpoints/',
        help="Path to a model checkpoint filename.",
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        required=False,
        default='data',
        help="Path to data.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for training"
    )

    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="Iterations for training."
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Number of training iterations which pass before evaluation."
    )

    parser.add_argument(
        "--eval_iters",
        type=int,
        default=10,
        help="Number of evaluation batches to estimate loss with."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Iterations for training."
    )

    parser.add_argument(
        "--save_iter",
        type=int,
        default=100,
        help="Iterations for model checkpointing."
    )


    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout probability during training."
    )

    parser.add_argument(
	    "--neurons_per_layer",
	    type=parse_layers,
	    required=True,
	    help="Sequence of number of neurons per layer, e.g., '4,8,16,8,4'. Check other necessary conditions in trainer.py"
    )

    parser.add_argument(
        "--standardize",
        type=int,
        default=1,
        help="1 = True, 0 = False, standardizes the data"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Sample size for chosen dataset (where applicable)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed."
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.3,
        help="Percentage of data used for validation"
    )

    parser.add_argument(
        "--intervention_var",
        type=int,
        default=10,
        help="List of variable names to perform an intervention and evaluate the causal effect."
    )

    parser.add_argument(
        "--run_optuna",
        type=int,
        default=0,
        help="Run hyperparameter search = 1 or not = 0."
    )

    parser.add_argument(
        "--optuna_num_trials",
        type=int,
        default=10,
        help="Number of optuna hyperparameter search trials."
    )
    parser.add_argument(
        "--checkpointing_on",
        type=int,
        default=0,
        help="Whether to save model checkpoints every save_iter."
    )

    parser.add_argument(
        "--shuffling",
        type=int,
        default=0,
        help="Whether to shuffle the ordering of variables and the dag, 0 = no, 1 = yes"
    )


    args = parser.parse_args()

    if args.run_optuna:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(lambda trial: objective(trial, args), n_trials=args.optuna_num_trials)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        objective(trial=None, args=args)



