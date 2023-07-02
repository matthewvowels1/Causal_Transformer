import argparse
from typing import Union, Dict, Optional
from pathlib import Path
from datasets import generate_data
import pandas as pd
import os
import numpy as np
import trainer
from model import CaT

def main(args):
	device = args.device
	dataset = args.dataset
	fn = args.data_path

	all_data, _ = generate_data(N=1000000, seed=args.seed, dataset=dataset)
	Y1, Y0 = all_data[:,-2], all_data[:,-1]
	ATE = (Y1 - Y0).mean()  # ATE based off a large sample

	all_data, DAG = generate_data(N=args.sample_size, seed=args.seed, dataset=dataset)

	df = pd.DataFrame(all_data)
	df.to_csv(os.path.join(fn, 'all_{}.csv'.format(dataset)), index=False)

	real_data = all_data[:, :-2]   # exclude the counterfactual columns Y1 and Y0
	num_vars = real_data.shape[1]

	model = CaT(num_vars=num_vars,
	            n_embed=args.n_embed,
	            dropout_rate=args.dropout_rate,
	            num_heads=args.num_heads,
	            dag=DAG,
	            n_layers=args.n_layers,
	            device=device).to(device)





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
		default='cpu',
		help="Device to train and/or run the model."
	)
	parser.add_argument(
		"--train",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Whether to train the model, if not, just test."
	)
	parser.add_argument(
		"--existing_model_path",
		type=Path,
		required=False,
		help="Path to a model checkpoint file (if not specified, will use random init).",
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
		default=32,
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
		"--learning_rate",
		type=int,
		default=3e-4,
		help="Iterations for training."
	)
	parser.add_argument(
		"--n_embed",
		type=int,
		default=64,
		help="Number of embedding dimensions"
	)

	parser.add_argument(
		"--num_heads",
		type=int,
		default=8,
		help="Number of transformer heads."
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

	args = parser.parse_args()

	main(args)

