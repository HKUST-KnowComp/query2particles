CUDA_VISIBLE_DEVICES=2 python run_q2b_benchmark.py \
--data_name "NELL-q2b" \
-b 1024 \
--model_name "v00.05" \
-p 2 \
-lr 0.0007 \
--warm_up_steps 1000 \
--dropout_rate 0.3 \
-wc 0 \
-ls 0.2