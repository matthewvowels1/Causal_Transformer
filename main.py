import argparse
from pathlib import Path
from datasets import generate_data
import pandas as pd
import os
import numpy as np
import trainer
from model import CaT
import torch
import matplotlib.pyplot as plt
import networkx as nx

def main(args):
	np.random.seed(seed=args.seed)
	torch.manual_seed(args.seed)
	device = args.device
	dataset = args.dataset
	fn = args.data_path

	_, _, _, _, Y0, Y1 = generate_data(N=1000000, seed=args.seed, dataset=dataset)
	ATE = (Y1 - Y0).mean()  # ATE based off a large sample

	all_data, DAG, causal_ordering, var_types, Y0, Y1 = generate_data(N=args.sample_size, seed=args.seed, dataset=dataset)

	DAG = nx.to_numpy_array(DAG)   # this will be the adjacency matrix

	# the last variable in the DAG should be the one that needs to be predicted
	emp_ATE = (Y1 - Y0).mean()  # ATE based off a small sample

	df = pd.DataFrame(all_data)
	df.to_csv(os.path.join(fn, 'all_{}.csv'.format(dataset)), index=False)

	indices = np.arange(0, len(all_data))
	np.random.shuffle(indices)

	val_inds = indices[:int(args.validation_fraction*len(indices))]
	train_inds = indices[int(args.validation_fraction*len(indices)):]
	train_data = all_data[train_inds]
	val_data = all_data[val_inds]

	print('Training data size:', train_data.shape, ' Validation data size:', val_data.shape)

	model = CaT(dropout_rate=args.dropout_rate,
	            head_size=args.head_size,
	            num_heads=args.num_heads,
	            dag=DAG,
	            n_layers=args.n_layers,
	            device=device,
	            var_types=var_types
	            ).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

	if args.existing_model_path == 'None':  # if no specified checkpoint file is given, train the model
		print('No existing model path specified, training from scratch for {} iterations.'.format(args.max_iters))
		trainer.train(train_data=train_data,
		              eval_iters=args.eval_iters,
		              val_data=val_data,
		              max_iters=args.max_iters,
		              eval_interval=args.eval_interval,
		              device=device,
		              model=model,
		              batch_size=args.batch_size,
		              save_iter=args.save_iter,
		              model_save_path=args.model_save_path,
		              optimizer=optimizer
		              )

	else:  # if existing checkpoint file is given, compare iteration against max_iters and finish training if necessary
		print('Loading checkpoint file at: ', args.existing_model_path)
		checkpoint = torch.load(args.existing_model_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		checkpoint_iter = checkpoint['iteration']

		if checkpoint_iter != args.max_iters:
			trainer.train(train_data=train_data,
			              eval_iters=args.eval_iters,
			              val_data=val_data,
			              max_iters=args.max_iters,
			              eval_interval=args.eval_interval,
			              device=device,
			              model=model,
			              batch_size=args.batch_size,
			              save_iter=args.save_iter,
			              model_save_path=args.model_save_path,
			              optimizer=optimizer, start_iter=checkpoint_iter
			              )



	maps = []
	for j in range(args.n_layers):
		heads = model.blocks[j].sa.heads
		for i in range(args.num_heads):
			maps.append(heads[i].wei.mean(0).cpu().detach().numpy())

	maps = np.stack(maps).mean(0)
	fig, ax = plt.subplots()
	im = ax.imshow(maps, cmap='hot', interpolation='nearest')
	cbar = ax.figure.colorbar(im, ax=ax, shrink=1)
	# Setting the axis tick labels
	ax.set_xticks(np.arange(len(list(DAG.nodes))))
	ax.set_yticks(np.arange(len(list(DAG.nodes))))

	ax.set_xticklabels(list(DAG.nodes))
	ax.set_yticklabels(list(DAG.nodes))

	# Rotating the tick labels inorder to set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	fig.tight_layout()
	plt.show()





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

	main(args)

