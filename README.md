# Causal_Transformer
CaT - Causal Transformer

TODO:
- [DONE] Make DAG a networkx object
- Change losses and predictions
- [DONE] based on the networkX object, find topological ordering of the variables
- [DONE] we need a variable type-based losses (e.g. BCE for binary vars, MSE for continuous vars) using the var_types variable
- Any variables with zero ordering can have their losses masked (can't predict something which has no parents)
- find a solution to deal with variables which have the same topological position in the causal ordering
- find a solution for intervening on multiple variables simultaneously (taking a list of intervention nodes)
- get the model to predict all positions in the causal ordering (as in a regular transformer)
- create a generate function which recursively predicts and feeds back into the model to generate predictions at any arbitrary position
- automatically derive ATE estimates for all paths by iterating through the structure
- Create testbed


## Example logic

For the following graph:

I1 -> X
I2 -> X
X -> M -> Y -> C
X -> Y

We construct the following (lower triangular) adjacency matrix:

----I1----I2----X----M----Y----C

I1---0----0-----0----0----0----0  ->  N/A (masked, 0th in causal ordering)

I2---0----0-----0----0----0----0  ->  N/A (masked, 0th in causal ordering)

X---1----1-----0----0----0----0   ->  X  (predict X)

M---0----0-----1----0----0----0   ->  M  (predict M)

Y---0----0-----1----1----0----0   ->  Y  (predict Y)

C---0----0-----0----0----1----0   ->  C  (predict C) 



Run using ```conda activate nlp_gpt_env```

checkpoints/model_10000_0.51.ckpt

python3 main.py --dataset general \
--max_iters 20000 \
--intervention_column 7  \
--validation_fraction 0.3 \
--sample_size 5000  \
--device cuda \
--existing_model_path  None \
--model_save_path checkpoints \
--data_path data \
--batch_size 32 \
--eval_interval 500 \
--eval_iters 100 \
--learning_rate 1e-4 \
--save_iter 2000 \
--num_heads 4 \
--head_size 4 \
--dropout_rate 0.3 \
--n_layers 4 

