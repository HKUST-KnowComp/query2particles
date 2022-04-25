CUDA_VISIBLE_DEVICES=1 python run_q2b_benchmark.py \
--data_name "FB15k-237-q2b" \
-b 2048 \
--model_name "v00.04" \
-p 2 \
-lr 0.001 \
--warm_up_steps 1000 \
--dropout_rate 0.2 \
-wc 0 \
-ls 0.2