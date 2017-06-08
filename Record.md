# Record

## CNN\_RNN
1. ./log/cnn\_rnn first completed try about cnn based attention rnn
   The config's details:
   1. Embedding file: vectors.bin
   * Embedding dim: 300
   * Max Sequence Length: 30
   * Validation split: 0.1
   * num rnn: 256
   * num dense: 200
   * rate drop rnn: 0.25
   * rate drop dense: 0.25
   * act: relu
   The model's details:
   1. input layer
   * embedding layer
   * Simple CNN layer
   * attention layer to embedding layer
   * BiGRU layer
   * multiply
   * Dropout
   * BatchNormalization
   * Dense
   * Dropout
   * BatchNormalization
   * Dense(sigmoid)

## basic\_baseline
1. ./log/basic\_baseline contains the first try about it
* ./log/basic\_baseline\_1: I try to re\_show the best score I get on kaggle. 
   The config's details:
   1. Embedding file: vectors.bin
   * Embedding dim: 300
   * Max Sequence Length: 30
   * Validation split: 0.1
   * num rnn: 256
   * num dense: 200
   * rate drop rnn: 0.25
   * rate drop dense: 0.25
   * act: relu
   The model's details:
   1. input layer
   * embedding layer
   * BiGRU layer
   * multiply
   * Dropout
   * BatchNormalization
   * Dense
   * Dropout
   * BatchNormalization
   * Dense(sigmoid)

## basic\_cnn
1. ./log/basic\_cnn: I try the kaggle cnn model
   The config's details:
   1. Embedding file: vectors.bin
   * Embedding dim: 300
   * Max Sequence Length: 30
   * Validation split: 0.1
   * num rnn: 256
   * num dense: 200
   * rate drop rnn: 0.25
   * rate drop dense: 0.2
   * act: relu
   The model's details:
   1. input layer
   * embedding layer
   * cnn layer: filters128, kernel\_size1, act relu
   * cnn layer: filters128, kernel\_size2, act relu
   * cnn layer: filters128, kernel\_size3, act relu
   * cnn layer: filters128, kernel\_size4, act relu
   * cnn layer: filters32, kernel\_size5, act relu
   * cnn layer: filters32, kernel\_size6, act relu
   * concatenate
   * diff
   * mul
   * concatenate
   * Dropout
   * BatchNormalization
   * Dense
   * Dropout
   * BatchNormalization
   * Dense(sigmoid)
