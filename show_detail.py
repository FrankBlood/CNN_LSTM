# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D
from keras.layers import Dropout, Input, Bidirectional, Merge, RepeatVector, Activation, TimeDistributed, Flatten, RepeatVector, Permute, Lambda
from keras.layers.merge import concatenate, add, dot, multiply
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
# from keras.utils import plot_model

# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'

########################################
## CNN based RNN
########################################
def cnn_rnn(nb_words=10000, EMBEDDING_DIM=200, \
            MAX_SEQUENCE_LENGTH=20, \
            num_lstm=200, num_dense=200, rate_drop_lstm=0.5, \
            rate_drop_dense=0.5, act='relu'):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)

    lstm_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    cnn_1 = Conv1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(embedded_sequences_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = Conv1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(cnn_1)
    cnn_1 = GlobalMaxPooling1D()(cnn_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = Dense(200)(cnn_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = BatchNormalization()(cnn_1)

    cnn_2 = Conv1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(embedded_sequences_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = Conv1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(cnn_2)
    cnn_2 = GlobalMaxPooling1D()(cnn_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = Dense(200)(cnn_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = BatchNormalization()(cnn_2)

    x1 = multiply([cnn_1, embedded_sequences_1])
    x1 = Activation('softmax')(x1)
    x1 = multiply([x1, embedded_sequences_1])

    x2 = multiply([cnn_2, embedded_sequences_2])
    x2 = Activation('softmax')(x2)
    x2 = multiply([x2, embedded_sequences_2])

    x1 = lstm_layer(x1)

    x2 = lstm_layer(x2)

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

    lstm_layer = Bidirectional(GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

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

if __name__ == '__main__':
    model = cnn_rnn()
    # model = basic_baseline()
