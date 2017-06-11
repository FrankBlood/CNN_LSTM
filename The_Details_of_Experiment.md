# 实验细节
## 数据集
* 我们使用 `Quora Question Pairs` 数据集: 共有404301对数据。 
* 预处理: 我们对所有的句子将大写转化为小写，去除标点，利用nltk的SnowballStemmer得到词根。
* 预训练[word2vec](https://code.google.com/archive/p/word2vec/): 我们基于本语料训练词向量，参数如下：`-cbow 0 -size 300 -window 5 -negative 0 -hs 1 -sample 1 e-3 -threads 12 -binary 1`

## Basic Baseline
* 我们利用BiGRU分别对两个句子建模，然后取最后一个step的隐藏层作为句子向量表达
* 结果：
| Epoch：12	| Train Loss：0.2209 | Train Acc：0.8431	| Dev Loss：0.3001	| Dev Acc：0.8479	| Test Loss：0.3310123806880918	| Test Acc：0.85018179986121234	|
* 代码如下:

```

    ########################################
    ## 共享层设计
    ########################################
    # Embedding层：利用预训练好的词向量，设置为在训练过程中参数不变。
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    # BiGRU层：num_rnn为256（也就是说隐藏层是256维的向量），dropout分别是0.25
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))

    ########################################
    ## 模型流程
    ########################################
    # 将输入从one-hot映射成embedding
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    # 由embedding作为输入，经过BiGRU的处理后，得到最后一个step的隐藏层向量作为句子的representation
    x1 = rnn_layer(embedded_sequences_1)

    x2 = rnn_layer(embedded_sequences_2)

    # 对两个句子的representation做multiply。即对应项相乘。
    merged = multiply([x1, x2])

    # 加Dropout（0.25）和BatchNormalization
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    # 加一层全连接（200维），激活函数为relu。然后加Dropout（0.25）和BatchNormalization
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    # 加一层输出层（1维），激活函数为sigmoid
    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## 训练模型
    ########################################
    # 输入为两个句子的one-hot表达，输出为两个句子是否matching的概率
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    
    # loss function用交叉熵；优化函数用nadam；评价指标用准确率acc
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model
```

## CNN_RNN
* 先利用CNN对句子进行表达，得到一个向量。然后把这个向量利用attention的方法对输入的embedding进行加权得到新的embedding，然后再输入到BiGRU中。之后的框架同basic baseline所示。（CNN的模型越复杂，在训练集的效果越好）
* 结果：
| Epoch：12	| Train Loss：0.2148 | Train Acc：0.8489	| Dev Loss：0.3009	| Dev Acc：0.8498	| Test Loss：0.33263399748324629	| Test Acc：0.85055282065279192	|
* 代码如下：

```

    ########################################
    ## 共享层设置
    ########################################
    # Embedding层：利用预训练好的词向量，设置为在训练过程中参数不变。
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    # BiGRU层：num_rnn为256（也就是说隐藏层是256维的向量），dropout分别是0.25
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))
    # Conv1D层： filters选择32个， kernel_size选择4个。
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=32, kernel_size=4)
    # 最大池化层
    pooling_layer = GlobalMaxPooling1D()
    # 全连接层1（用来接池化层的）
    cnn_dense = Dense(300)
    # 全连接层2（attention中用：用来将CNN的表达转化为和embedding相同维数的映射的）
    cnn_dense1 = Dense(300)
    # 防止过拟合的
    cnn_dropout1 = Dropout(0.2)
    cnn_dropout2 = Dropout(0.2)
    cnn_batchnormalization = BatchNormalization()
    
    ########################################
    ## 模型流程
    ########################################

    # 输入从one-hot变成embedding
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    # 分别得到两个句子的CNN表达
    cnn_1 = cnn_layer(embedded_sequences_1)
    cnn_1 = pooling_layer(cnn_1)
    cnn_1 = cnn_dropout1(cnn_1)
    cnn_1 = cnn_dense(cnn_1)
    cnn_1 = cnn_dropout2(cnn_1)
    cnn_1 = cnn_batchnormalization(cnn_1)

    cnn_2 = cnn_layer(embedded_sequences_2) 
    cnn_2 = cnn_layer1(cnn_2)
    cnn_2 = pooling_layer(cnn_2)
    cnn_2 = cnn_dropout1(cnn_2)
    cnn_2 = cnn_dense(cnn_2)
    cnn_2 = cnn_dropout2(cnn_2)
    cnn_2 = cnn_batchnormalization(cnn_2)

    # attention第一步，把CNN的表达转化为和embedding同维的向量
    cnn_1_t = cnn_dense1(cnn_1)
    cnn_2_t = cnn_dense1(cnn_2)

    # 计算转化后的向量和原embedding的内积
    a1 = multiply([cnn_1_t, embedded_sequences_1])
    a2 = multiply([cnn_2_t, embedded_sequences_2])
    
    # Permute的作用是矩阵转置
    a1 = Permute([2, 1])(a1)
    a2 = Permute([2, 1])(a2)
    
    # 上面是两个向量的对应相乘，现在做相加，得到内积
    a1 = Lambda(lambda x: K.sum(x, axis=1))(a1)
    a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)

    # 得到内积后加sigmoid，作为attention的输出，即得到每个embedding的权重
    a1 = Activation('sigmoid')(a1)
    a2 = Activation('sigmoid')(a2)

    # 把attention的权重乘到每个embedding上，得到新的embedding
    embedded_sequences_1 = Permute([2, 1])(embedded_sequences_1)
    embedded_sequences_2 = Permute([2, 1])(embedded_sequences_2)
    
    x1 = multiply([a1, embedded_sequences_1])
    x2 = multiply([a2, embedded_sequences_2])

    x1 = Permute([2, 1])(x1)
    x2 = Permute([2, 1])(x2)

    # 得到的新的embedding输入BiGRU中，输出的最后一个step的隐藏层向量作为句子的representation。
    x1 = rnn_layer(x1)
    x2 = rnn_layer(x2)

    # 对两个句子的representation做multiply。即对应项相乘。
    merged = multiply([x1, x2])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    # 加一层全连接（200维），激活函数为relu。然后加Dropout（0.25）和BatchNormalization
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    # 加一层输出层（1维），激活函数为sigmoid
    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## 模型训练
    ########################################
    # 输入为两个句子的one-hot表达，输出为两个句子是否matching的概率
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    # loss function用交叉熵；优化函数用nadam；评价指标用准确率acc
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    return model
```