# CNN_LSTM
* add CNN as an attention to LSTM

# Summary
| Model\_Log | Epoch | Train Loss | Train Acc | Dev Loss | Dev Acc | Test Loss | Test Acc | Model Weight |
| :----------|:-----:| :---------:| :-------: | :------: | :-----: | :-------: | :------: | :----------: |
| ./log/basic\_baseline\_2 | 12 | 0.2209 | 0.8431	| 0.3001 | 0.8479 | 0.3310123806880918 | 0.85018179986121234 | basic_baseline_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5 |
| ./log/cnn\_rnn\_1 | 12 | 0.2148 | 0.8489 | 0.3009 | 0.8498 | 0.33263399748324629 | 0.85055282065279192 | cnn_rnn_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5 |

# Record

## CNN\_RNN
1. ./new\_log/cnn\_rnn: 
   The config's details:
   > * Embedding file: vectors.bin
   > * Embedding dim: 300
   > * Max Sequence Length: 30
   > * Validation split: 0.2
   > * num rnn: 256
   > * num dense: 200
   > * rate drop rnn: 0.25
   > * rate drop dense: 0.25
   > * act: relu
   > * dev 0.1, test 0.1

   The model's details:
   > * input layer
   > * embedding layer: trainable=True
   > * CNN layer: 64 filters and 3 kernels
   > * attention layer to embedding layer
   > * BiGRU layer(shown as config)
   > * multiply
   > * Dropout
   > * BatchNormalization
   > * Dense
   > * Dropout
   > * BatchNormalization
   > * Dense(sigmoid)

## basic\_baseline
1. ./new\_log/basic\_baseline
   The config's details:
   > 1. Embedding file: vectors.bin
   > 2. Embedding dim: 300
   > 3. Max Sequence Length: 30
   > 4. Validation split: 0.2
   > 5. num rnn: 256
   > 6. num dense: 200
   > 7. rate drop rnn: 0.25
   > 8. rate drop dense: 0.25
   > 9. act: relu
   > 10. dev 0.1, test 0.1

   The model's details:
   > 1. input layer
   > 2. embedding layer: trainable=True
   > 3. BiGRU layer(shown as config)
   > 4. multiply
   > 5. Dropout
   > 6. BatchNormalization
   > 7. Dense
   > 8. Dropout
   > 9. BatchNormalization
   > 10. Dense(sigmoid)

# To be continued...
## Task
Deep Semantic Matching

## Dataset
Quora Question Pairs

## Common LSTM
one layer LSTM

## CNN-based LSTM
add CNN as an attention
