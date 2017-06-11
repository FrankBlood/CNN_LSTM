# CUDA_VISIBLE_DEVICES=0 nohup python train.py 0 > ./log/basic_baseline_2 &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 1 > ./log/basic_attention_2 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py 2 > ./log/cnn_rnn_3 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py 3 > ./log/cnn_rnn_tmp_4 &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 4 > ./log/basic_cnn &
