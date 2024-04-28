# Causal_Transformer
CaT - Causal Transformer

TODO:
- [DONE] Make DAG a networkx object
- [DONE] Write function which tracks per variable loss
- [DONE] plot per variable loss
- [DONE] based on the networkX object, find topological ordering of the variables
- [DONE] we need a variable type-based losses (e.g. BCE for binary vars, MSE for continuous vars) using the var_types variable
- [DONE] Any variables with zero ordering can have their losses masked (can't predict something which has no parents)
- [DONE] find a solution to deal with variables which have the same topological position in the causal ordering
- [DONE] find a solution for intervening on multiple variables simultaneously (taking a list of intervention nodes)
- [DONE] get the model to predict all positions in the causal ordering (as in a regular transformer)
- [DONE] create a generate function which recursively predicts and feeds back into the model to generate predictions at any arbitrary position
- [DONE] create a very simple dataset for validation
- [DONE] turn into optuna objective
- [DONE] if each row is independent, then can we shuffle the variable order and construct an adjacency matrix based on the parents of each variable in each sample in each batch?
- [DONE] evaluate R2 to check convergence (not for final code)
- IDEA: Train with full lower triangular + diagonal in 1st stage of training (to help gradient flow), then drop out links
- Create testbed
  - compute all causal effects for 'general' dataset in datasets.py 
  - create a function for computing ATEs using CaT 
  - check that they can be estimated reasonably well
  - find application tasks / benchmarks (possibly robotics)
  - run test suite and report results


## Example logic

For the following graph:

I1 -> X

I2 -> X

X -> M -> Y -> C

X -> Y

We construct the following (lower triangular, topologically sorted) adjacency matrix:

----I1----I2----X----M----Y----C

I1---0----0-----0----0----0----0  ->  N/A (masked, 0th in causal ordering)

I2---0----0-----0----0----0----0  ->  N/A (masked, 0th in causal ordering)

X ---1----1-----0----0----0----0   ->  X  (predict X)

M ---0----0-----1----0----0----0   ->  M  (predict M)

Y ---0----0-----1----1----0----0   ->  Y  (predict Y)

C ---0----0-----0----0----1----0   ->  C  (predict C) 




## CLI Example
Run using ```conda activate nlp_gpt_env```

checkpoints/model_10000_0.51.ckpt

python3 main.py --dataset simple_test \
--max_iters 30000 \
--validation_fraction 0.3 \
--sample_size 5000  \
--device cuda \
--existing_model_path  None \
--model_save_path checkpoints \
--data_path data \
--batch_size 1 \
--eval_interval 500 \
--eval_iters 100 \
--learning_rate 1e-4 \
--checkpointing_on 0 \
--save_iter 100000 \
--num_heads 10 \
--head_size 1 \
--dropout_rate 0.45 \
--n_layers 2 \
--run_optuna 1 \
--optuna_num_trials 50 \
--score_interaction_type 1 \
--shuffling 0 \
--dag_type 2 \
--seed 2

