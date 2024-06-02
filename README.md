# Causal_Transformer
CaT - Causal Transformer

Update:

layernorm in CaT : Nope!

CFCN is working
CaT is working

TODO: Check mediated version DONE
TODO: Add loss mask for exogenous variables DONE
TODO: Talk about loss mask in paper
TODO: Finalize tutorials
TODO: Testbeds
TODO: CFCN NOT WORKING?

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

