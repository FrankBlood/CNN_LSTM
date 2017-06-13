# CNN_LSTM
* `model.py`: all models
* `train.py`: train details
* `config.py`: some global parameters
* `show_details.py`: debug the framework of models
* `run.sh`: run the train

# Summary
| Model\_Log | Epoch | Train Loss | Train Acc | Dev Loss | Dev Acc | Test Loss | Test Acc | Model Weight |
| :----------|:-----:| :---------:| :-------: | :------: | :-----: | :-------: | :------: | :----------: |
| ./log/basic\_baseline\_2 | 12 | 0.2209 | 0.8431	| 0.3001 | 0.8479 | 0.3310123806880918 | 0.85018179986121234 | basic_baseline_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5 |
| ./log/cnn\_rnn\_1 | 12 | 0.2148 | 0.8489 | 0.3009 | 0.8498 | 0.33263399748324629 | 0.85055282065279192 | cnn_rnn_256_200_0.25_0.25_Fri_Jun__9_06:17:01_2017.h5 |
| -- | -- | -- | -- | -- | -- | -- | -- | --|
| ./new\_log/basic\_baseline | 6 | 0.1129 | 0.9287 | 0.3980 | 0.8474 | 0.38274466106547178 | 0.83897697165416374 | basic_baseline_256_200_0.25_0.25_Mon_Jun_12_05:45:20_2017.h5 |
| ./new\_log/basic\_baseline\_1 | 12 | 0.2249 | 0.8399 | 0.2800 | 0.8508 | 0.33469125888022633 | 0.84568008038683273 | basic_baseline_256_200_0.25_0.25_Mon_Jun_12_06:33:15_2017.h5 |
| ./new\_log/basic\_baseline\_2 | 12 | 0.2126 | 0.8499 | 0.2905 | 0.8535 | 0.32628323810153453 | 0.85342204782775211 | basic_baseline_256_200_0.25_0.25_Mon_Jun_12_10:31:45_2017.h5|
| ./new\_log/cnn\_rnn | 7 | 0.1266 | 0.9183 | 0.4012 | 0.8458 | 0.39586822402159111 | 0.83788864403051233 | cnn_rnn_256_200_0.25_0.25_Mon_Jun_12_05:58:06_2017.h5 |
| ./new\_log/cnn\_rnn\_1 | 10 | 0.2224 | 0.8442 | 0.3048 | 0.8440 | 0.36778828029997723 | 0.83229859719933019 | cnn_rnn_256_200_0.25_0.25_Mon_Jun_12_07:55:21_2017.h5|
| ./new\_log/cnn\_rnn\_2 | 14 | 0.2318 | 0.8347 | 0.2926 | 0.8467 | 0.33884965215525081 | 0.84513591658532705 | cnn_rnn_256_200_0.25_0.25_Mon_Jun_12_08:50:42_2017.h5 |
| ./new\_log/cnn\_rnn\_3 | 9  | 0.2271 | 0.8396 | 0.3121 | 0.8317 | 0.355499757804914 | 0.84031264622255275  | cnn_rnn_256_200_0.25_0.25_Mon_Jun_12_10:42:58_2017.h5 |

# Record

## CNN\_RNN
1. ./new\_log/cnn\_rnn: 1 cnn layer filters 64, kernel 3 trainable
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

1. ./new\_log/cnn\_rnn\_1: 1 cnn layer filters 64, kernel 3 un-trainable
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
   > * embedding layer: trainable=False
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

1. ./new\_log/cnn\_rnn\_2: 1 cnn layer filters 64, kernels 2 un-trainable
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
   > * embedding layer: trainable=False
   > * CNN layer: 64 filters and 2 kernels
   > * attention layer to embedding layer
   > * BiGRU layer(shown as config)
   > * multiply
   > * Dropout
   > * BatchNormalization
   > * Dense
   > * Dropout
   > * BatchNormalization
   > * Dense(sigmoid)

1. ./new\_log/cnn\_rnn\_3: 1 cnn layer filters 128, kernels 2, un-trainable, monitor=val_acc
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
   > * embedding layer: trainable=False
   > * CNN layer: 128 filters and 2 kernels
   > * attention layer to embedding layer
   > * BiGRU layer(shown as config)
   > * multiply
   > * Dropout
   > * BatchNormalization
   > * Dense
   > * Dropout
   > * BatchNormalization
   > * Dense(sigmoid)

1. ./new\_log/cnn\_rnn\_4: 1 cnn layer filters 128, kernels 2, un-trainable, monitor=val_loss
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
   > * embedding layer: trainable=False
   > * CNN layer: 128 filters and 2 kernels
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
1. ./new\_log/basic\_baseline: trainable
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

1. ./new\_log/basic\_baseline\_1: untrainable
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
   > * embedding layer: trainable=False
   > * BiGRU layer(shown as config)
   > * multiply
   > * Dropout
   > * BatchNormalization
   > * Dense
   > * Dropout
   > * BatchNormalization
   > * Dense(sigmoid)

1. ./new\_log/basic\_baseline\_2: untrainable moniter=val_acc
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
   > * modelpoint: moniter = val_vcc

   The model's details:
   > * input layer
   > * embedding layer: trainable=False
   > * BiGRU layer(shown as config)
   > * multiply
   > * Dropout
   > * BatchNormalization
   > * Dense
   > * Dropout
   > * BatchNormalization
   > * Dense(sigmoid)

# To be continued...
## Task
Deep Semantic Matching

## Dataset
Quora Question Pairs

## Common LSTM
one layer LSTM

## CNN-based LSTM
add CNN as an attention
