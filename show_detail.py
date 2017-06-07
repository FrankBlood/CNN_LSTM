# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D, ConvLSTM2D
from keras.layers import Dropout, Input, Bidirectional, Merge, RepeatVector, Activation, TimeDistributed, Flatten, RepeatVector, Permute, Lambda
from keras.layers.merge import concatenate, add, dot, multiply
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

########################################
## CNN based RNN
########################################
def cnn_rnn(nb_words=10000, EMBEDDING_DIM=300, \
            MAX_SEQUENCE_LENGTH=50, \
            num_rnn=300, num_dense=300, rate_drop_rnn=0.25, \
            rate_drop_dense=0.25, act='relu'):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=32, kernel_size=4)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(300)
    cnn_dropout1 = Dropout(0.2)
    cnn_dropout2 = Dropout(0.2)
    cnn_batchnormalization = BatchNormalization()
    cnn_repeatvector = RepeatVector(MAX_SEQUENCE_LENGTH)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    cnn_1 = cnn_layer(embedded_sequences_1)
    cnn_1 = pooling_layer(cnn_1)
    cnn_1 = cnn_dropout1(cnn_1)
    cnn_1 = cnn_dense(cnn_1)
    cnn_1 = cnn_dropout2(cnn_1)
    cnn_1 = cnn_batchnormalization(cnn_1)

    cnn_2 = cnn_layer(embedded_sequences_2)    
    cnn_2 = pooling_layer(cnn_2)
    cnn_2 = cnn_dropout1(cnn_2)
    cnn_2 = cnn_dense(cnn_2)
    cnn_2 = cnn_dropout2(cnn_2)
    cnn_2 = cnn_batchnormalization(cnn_2)
    
    cnn_1 = cnn_repeatvector(cnn_1)
    cnn_2 = cnn_repeatvector(cnn_2)

    a1 = multiply([cnn_1, embedded_sequences_1])
    a2 = multiply([cnn_2, embedded_sequences_2])
    
    a1 = Permute([2, 1])(a1)
    a2 = Permute([2, 1])(a2)
    
    a1 = Lambda(lambda x: K.sum(x, axis=1))(a1)
    a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)
    
    embedded_sequences_1 = Permute([2, 1])(embedded_sequences_1)
    embedded_sequences_2 = Permute([2, 1])(embedded_sequences_2)
    
    x1 = multiply([a1, embedded_sequences_1])
    x2 = multiply([a2, embedded_sequences_2])

    x1 = Permute([2, 1])(x1)
    x2 = Permute([2, 1])(x2)

    x1 = rnn_layer(x1)
    x2 = rnn_layer(x2)

    merged = multiply([x1, x2])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    # x1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_1)
    # x1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(x1)

    # y1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_2)
    # y1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(y1)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## basic baseline
########################################
def basic_baseline(nb_words=10000, EMBEDDING_DIM=200, \
            MAX_SEQUENCE_LENGTH=20, \
            num_lstm=200, num_dense=200, rate_drop_lstm=0.5, \
            rate_drop_dense=0.5, act='relu'):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)

    lstm_layer = Bidirectional(ConvLSTM2D(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    x1 = lstm_layer(embedded_sequences_1)

    x2 = lstm_layer(embedded_sequences_2)

    merged = multiply([x1, x2])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    # x1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_1)
    # x1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(x1)

    # y1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_2)
    # y1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(y1)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

########################################
## basic attention
########################################
def basic_attention(nb_words=10000, EMBEDDING_DIM=300, \
                    MAX_SEQUENCE_LENGTH=40, \
                    num_rnn=300, num_dense=300, rate_drop_rnn=0.25, \
                    rate_drop_dense=0.25, act='relu'):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn, return_sequences=True))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    print(embedded_sequences_1.shape)
    x1 = rnn_layer(embedded_sequences_1)
    print(x1.shape)

    # attention1 = TimeDistributed(Dense(1, activation='tanh'))(x1)
    attention1 = Dense(40, activation='tanh')(x1)
    print(attention1.shape)
    attention1 = Dense(40, activation='softmax')(attention1)
    print(attention1.shape)
    attention1 = multiply([x1, attention1])
    print(attention1.shape)
    x1 = Lambda(lambda x: K.sum(x, axis=1))(attention)
    print(x1.shape)

    # attention1 = Flatten()(attention1)
    # attention1 = Activation('softmax')(attention1)
    # attention1 = RepeatVector(num_rnn)(attention1)
    # attention1 = Permute([2, 1])(attention1)
    # attention1 = multiply([x1, attention1])
    # x1 = Lambda(lambda xin: K.sum(xin, axis=1))(attention1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = rnn_layer(embedded_sequences_2)
    attention2 = TimeDistributed(Dense(1, activation='tanh'))(y1)
    attention2 = Flatten()(attention2)
    attention2 = Activation('softmax')(attention2)
    attention2 = RepeatVector(num_rnn)(attention2)
    attention2 = Permute([2, 1])(attention2)
    attention2 = multiply([y1, attention2])
    y1 = Lambda(lambda xin: K.sum(xin, axis=1))(attention2)

    merged = multiply([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

if __name__ == '__main__':
    model = cnn_rnn()
    # model = basic_attention()
    # model = basic_baseline()
    # plot_model(model, to_file='model.png', show_shapes=True)
