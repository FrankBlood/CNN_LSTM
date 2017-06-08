# Record

## CNN\_RNN
1. log as cnn\_rnn\_1, the detail of model as summary shows.
2. basic baseline as basic\_baseline
3. basic attention as basic\_attention

## basic\_baseline
1. ./log/basic\_baseline contains the first try about it
2. ./log/basic\_baseline\_1: I try to re\_show the best score I get on kaggle. 
   The config's details:
   1. Embedding file: vectors.bin
   2. Embedding dim: 300
   3. Max Sequence Length: 30
   4. Validation split: 0.1
   5. num rnn: 256
   6. num dense: 200
   7. rate drop rnn: 0.25
   8. rate drop dense: 0.25
   9. act: relu
   The model's details:
   1. input layer
   2. embedding layer
   3. BiGRU layer
   4. multiply
   5. Dropout
   6. BatchNormalization
   7. Dense
   8. Dropout
   9. BatchNormalization
   10. Dense(sigmoid)
