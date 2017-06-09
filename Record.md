# Summary
| Model\_Log | Epoch | Train Loss | Train Acc | Dev Loss | Dev Acc | Test Loss | Test Acc | Model Weight |
| :----------|:-----:| :---------:| :-------: | :------: | :-----: | :-------: | :------: | :----------: |
| basic\_baseline | 6 | 0.2937 | 0.7740 | 0.3497 | 0.8225 | Nan | Nan | basic_baseline_256_200_0.25_0.25.h5 |
| basic\_baseline\_1 | 8 | 0.2279 | 0.8357 | 0.2803 | 0.8546 | Nan | Nan | basic_baseline_256_200_0.25_0.25_Thu_Jun__8_08:57:23_2017.h5 |
| basic\_baseline\_2 | 12 | 0.2209 | 0.8431 | 0.3001 | 0.8479 | 0.3310123806880918 | 0.85018179986121234 | basic_baseline_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5|
| basic\_attention | 5 | 0.2729 | 0.7827 | 0.2904 | 0.8303 | Nan | Nan | basic_attention_256_200_0.25_0.25.h5 |
| basic\_attention\_1 | 8 | 0.2270 | 0.8349 | 0.2889 | 0.8552 | Nan | Nan | basic_attention_256_200_0.25_0.25_Thu_Jun__8_11:14:25_2017.h5 |
| basic\_cnn | 4 | 0.1504 | 0.9032 | 0.2585 | 0.8636 | Nan | Nan | basic_cnn_256_200_0.25_0.25_Thu_Jun__8_10:26:04_2017.h5 |
| cnn\_rnn | 9 | 0.2241 | 0.8415 | 0.2859 | 0.8532 | Nan | Nan | cnn_rnn_256_200_0.25_0.25_Thu_Jun__8_09:38:51_2017.h5 |
| cnn\_rnn\_1 | 12 | 0.2148 | 0.8489 | 0.3009 | 0.8498 | 0.33263399748324629 | 0.85055282065279192 | cnn_rnn_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5|
| cnn\_rnn\_tmp | 6 | 0.1408 | 0.9125 | 0.3441 | 0.8341 | Nan | Nan | cnn_rnn_tmp_256_200_0.25_0.25_Thu_Jun__8_11:14:25_2017.h5 |

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

3. ./log/cnn\_rnn\_1: same as the first one but set 0.1 for dev and 0.1 for test
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
   10. 0.1 dev and 0.1 test
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

4. ./log/cnn\_rnn\_tmp\_2: same as the second one but set 0.1 for dev and 0.1 for test
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
   10. 0.1 dev and 0.1 test
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
3. ./log/basic\_attention\_2: I try to re\_show the same config of my best score I got on kaggle.
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
   10. 0.1 dev and 0.1 test
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
