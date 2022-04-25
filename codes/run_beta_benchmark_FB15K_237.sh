CUDA_VISIBLE_DEVICES=1 python run_beta_benchmark.py \
--data_name "FB15k-237-betae" \
-b 8192 \
--model_name "v00.01" \
-p 2 \
-lr 0.001 \
--warm_up_steps 1000 \
--dropout_rate 0.1 \
-wc 0 \
-ls 0.5