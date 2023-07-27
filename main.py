import argparse
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from trainer import objective


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='synth1',
        required=True,
        help="Name of the dataset to be used for training and/or testing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
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
        "--num_heads",
        type=int,
        default=4,
        help="Number of transformer heads."
    )

    parser.add_argument(
        "--head_size",
        type=int,
        default=4,
        help="Embedding dimension within the transformer."
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout probability during training."
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=8,
        help="Number of layers in model."
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
        default=1000,
        help="Sample size for chosen dataset (where applicable)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=20)
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


