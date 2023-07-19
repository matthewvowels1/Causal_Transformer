# Causal_Transformer
CaT - Causal Transformer

TODO:
- [DONE] Make DAG a networkx object
- Change losses and predictions
- [DONE] based on the networkX object, find topological ordering of the variables
- we need a variable type-based losses (e.g. BCE for binary vars, MSE for continuous vars) using the var_types variable
- find a solution to deal with variables which have the same topological position in the causal ordering
- get the model to predict all positions in the causal ordering (as in a regular transformer)
- create a generate function which recursively predicts and feeds back into the model to generate predictions at any arbitrary position
- automatically derive ATE estimates for all paths by iterating through the structure
- Create testbed


Run using ```conda activate nlp_gpt_env```

checkpoints/model_10000_0.51.ckpt

python3 main.py --dataset general \
--max_iters 20000 \
--intervention_column 7  \
--effect_column 8 \
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
--n_layers 4 \
--no-continuous_outcome
