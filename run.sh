# CUDA_VISIBLE_DEVICES=0 nohup python train.py 0 > ./log/basic_baseline &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 1 > ./log/basic_attention &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 2 > ./log/cnn_rnn &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py 3 > ./log/cnn_rnn_tmp &
CUDA_VISIBLE_DEVICES=0 nohup python train.py 4 > ./log/basic_cnn &
