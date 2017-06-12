# CNN_LSTM
* add CNN as an attention to LSTM

# Summary
| Model\_Log | Epoch | Train Loss | Train Acc | Dev Loss | Dev Acc | Test Loss | Test Acc | Model Weight |
| :----------|:-----:| :---------:| :-------: | :------: | :-----: | :-------: | :------: | :----------: |
| basic\_baseline | 6 | 0.2937 | 0.7740 | 0.3497 | 0.8225 | Nan | Nan | basic_baseline_256_200_0.25_0.25.h5 |

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
