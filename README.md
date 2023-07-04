# Causal_Transformer
CaT - Causal Transformer

TODO: sort out causal ordering so that the mask can be used to predict all intermediate variables

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
