CUDA_VISIBLE_DEVICES=2 python run_beta_benchmark.py \
--data_name "NELL-betae" \
-b 3000 \
--model_name "v00.02" \
-p 2 \
-lr 0.0007 \
--warm_up_steps 1000 \
--dropout_rate 0.3 \
-wc 0 \
-ls 0.7