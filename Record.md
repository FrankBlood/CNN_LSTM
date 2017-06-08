# Record

## CNN\_RNN
1. ./log/cnn\_rnn first completed try about cnn based attention rnn
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
   3. Simple CNN layer
   4. attention layer to embedding layer
   5. BiGRU layer
   6. multiply
   7. Dropout
   8. BatchNormalization
   9. Dense
   10. Dropout
   11. BatchNormalization
   12. Dense(sigmoid)
2. ./log/cnn\_rnn\_tmp: second completed about another cnn based attention rnn
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
   3. Complex CNN layer
   4. attention layer to embedding layer
   5. BiGRU layer
   6. multiply
   7. Dropout
   8. BatchNormalization
   9. Dense
   10. Dropout
   11. BatchNormalization
   12. Dense(sigmoid)

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

## basic\_attention
1. ./log/basic\_attention contains the first try about it
2. ./log/basic\_attention\_1: I try to re\_show the same config of my best score I got on kaggle.
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
   5. self-attention
   6. Dropout
   7. BatchNormalization
   8. Dense
   9. Dropout
   10. BatchNormalization
   11. Dense(sigmoid)

## basic\_cnn
1. ./log/basic\_cnn: I try the kaggle cnn model
   The config's details:
   1. Embedding file: vectors.bin
   2. Embedding dim: 300
   3. Max Sequence Length: 30
   4. Validation split: 0.1
   5. num rnn: 256
   6. num dense: 200
   7. rate drop rnn: 0.25
   8. rate drop dense: 0.2
   9. act: relu
   The model's details:
   1. input layer
   2. embedding layer
   3. cnn layer: filters128, kernel\_size1, act relu
   4. cnn layer: filters128, kernel\_size2, act relu
   5. cnn layer: filters128, kernel\_size3, act relu
   6. cnn layer: filters128, kernel\_size4, act relu
   7. cnn layer: filters32, kernel\_size5, act relu
   8. cnn layer: filters32, kernel\_size6, act relu
   9. concatenate
   10. diff
   11. mul
   12. concatenate
   13. Dropout
   14. BatchNormalization
   15. Dense
   16. Dropout
   17. BatchNormalization
   18. Dense(sigmoid)
