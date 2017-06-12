# CUDA_VISIBLE_DEVICES=0 nohup python train.py 0 > ./new_log/basic_baseline &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 1 > ./new_log/basic_attention &
CUDA_VISIBLE_DEVICES=1 nohup python train.py 2 > ./new_log/cnn_rnn &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py 3 > ./new_log/cnn_rnn_tmp &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 4 > ./new_log/basic_cnn &
