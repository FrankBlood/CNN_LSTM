# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Convolution1D, GlobalMaxPooling1D
from keras.layers import Dropout, Input, Activation, Flatten
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Merge
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
def cnn_rnn(nb_words, EMBEDDING_DIM, \
            embedding_matrix, MAX_SEQUENCE_LENGTH, \
            num_rnn, num_dense, rate_drop_rnn, \
            rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    cnn_1 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)(embedded_sequences_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)(cnn_1)

    cnn_1 = GlobalMaxPooling1D()(cnn_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = Dense(300)(cnn_1)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = BatchNormalization()(cnn_1)

    cnn_2 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)(embedded_sequences_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)(cnn_2)
    
    cnn_2 = GlobalMaxPooling1D()(cnn_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = Dense(300)(cnn_2)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = BatchNormalization()(cnn_2)
    
    print cnn_1.shape
    print cnn_2.shape
    print embedded_sequences_1.shape
    print embedded_sequences_2.shape

    x1 = TimeDistributed(Lambda(lambda x: dot([x, cnn_1], 1)))(embedded_sequences_1)
    x1 = Activation('softmax')(x1)
    x1 = multiply([x1, embedded_sequences_1])

    x2 = TimeDistributed(Lambda(lambda x: dot([x, cnn_2], 1)))(embedded_sequences_2)
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
def basic_baseline(nb_words, EMBEDDING_DIM, \
                   embedding_matrix, MAX_SEQUENCE_LENGTH, \
                   num_rnn, num_dense, rate_drop_rnn, \
                   rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))

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


# ########################################
# ## GRU attention multiply
# ########################################
# def gru_attention_multiply(nb_words, EMBEDDING_DIM, \
#            embedding_matrix, MAX_SEQUENCE_LENGTH, \
#            num_lstm, num_dense, rate_drop_lstm, \
#            rate_drop_dense, act):
#     embedding_layer = Embedding(nb_words,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)
#     lstm_layer = GRU(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

#     sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_1 = embedding_layer(sequence_1_input)
#     x1 = lstm_layer(embedded_sequences_1)
#     attention1 = TimeDistributed(Dense(1, activation='tanh'))(x1)
#     attention1 = Flatten()(attention1)
#     attention1 = Activation('softmax')(attention1)
#     attention1 = RepeatVector(num_lstm)(attention1)
#     attention1 = Permute([2, 1])(attention1)
#     attention1 = multiply([x1, attention1])
#     x1 = Lambda(lambda xin: K.sum(xin, axis=1))(attention1)

#     sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_2 = embedding_layer(sequence_2_input)
#     y1 = lstm_layer(embedded_sequences_2)
#     attention2 = TimeDistributed(Dense(1, activation='tanh'))(y1)
#     attention2 = Flatten()(attention2)
#     attention2 = Activation('softmax')(attention2)
#     attention2 = RepeatVector(num_lstm)(attention2)
#     attention2 = Permute([2, 1])(attention2)
#     attention2 = multiply([y1, attention2])
#     y1 = Lambda(lambda xin: K.sum(xin, axis=1))(attention2)

#     merged = multiply([x1, y1])
#     merged = Dropout(rate_drop_dense)(merged)
#     merged = BatchNormalization()(merged)

#     merged = Dense(num_dense, activation=act)(merged)
#     merged = Dropout(rate_drop_dense)(merged)
#     merged = BatchNormalization()(merged)

#     preds = Dense(1, activation='sigmoid')(merged)

#     ########################################
#     ## train the model
#     ########################################
#     model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
#     model.compile(loss='binary_crossentropy',
#               optimizer='nadam',
#               metrics=['acc'])
#     model.summary()
#     # print(STAMP)
#     return model

# ########################################
# ## GRU distance
# ########################################

# def euclidean_distance(vects):
#     x, y = vects
#     return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# def eucl_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)


# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     return K.mean(y_true * K.square(y_pred) +  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# def gru_distance(nb_words, EMBEDDING_DIM, \
#            embedding_matrix, MAX_SEQUENCE_LENGTH, \
#            num_lstm, num_dense, rate_drop_lstm, \
#            rate_drop_dense, act):
#     embedding_layer = Embedding(nb_words,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)
#     lstm_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

#     sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_1 = embedding_layer(sequence_1_input)
#     x1 = lstm_layer(embedded_sequences_1)
#     x1 = Dropout(rate_drop_lstm)(x1)
#     x1 = BatchNormalization()(x1)

#     sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_2 = embedding_layer(sequence_2_input)
#     y1 = lstm_layer(embedded_sequences_2)
#     y1 = Dropout(rate_drop_lstm)(y1)
#     y1 = BatchNormalization()(y1)

#     distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x1, y1])

#     ########################################
#     ## train the model
#     ########################################
#     model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=distance)
#     model.compile(loss=contrastive_loss,
#                   optimizer='nadam',
#                   metrics=['acc'])
#     model.summary()
#     # print(STAMP)
#     return model