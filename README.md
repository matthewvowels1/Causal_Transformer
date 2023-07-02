# Causal_Transformer
CaT - Causal Transformer


Run using ```conda activate nlp_gpt_env```


python3 main.py --dataset general \
--max_iters 1000 \
--intervention_column 10  \
--validation_fraction 0.3 \
--sample_size 5000  \
--device cuda \
--existing_model_path None \
--model_save_path checkpoints \
--data_path data \
--batch_size 32 \
--eval_interval 50 \
--eval_iters 50 \
--learning_rate 1e-4 \
--save_iter 500 \
--num_heads 8 \
--head_size 8 \
--dropout_rate 0.3 \
--n_layers 8 
