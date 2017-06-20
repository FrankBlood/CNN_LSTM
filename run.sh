# CUDA_VISIBLE_DEVICES=0 nohup python train.py 0 > ./tmp_log/basic_baseline &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 1 > ./log_for_100lstm/basic_attention &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 2 > ./tmp_log/cnn_rnn_6 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py 3 > ./tmp_log/cnn_rnn_tmp &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 4 > ./tmp_log/basic_cnn &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py 5 > ./tmp_log/cnn_rnn_add &
