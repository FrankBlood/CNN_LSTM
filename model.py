# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D, GlobalAveragePooling1D
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
    '''
    This is the basic cnn rnn model 

    model: input layer; embedding layer; cnn based attention layer; rnn layer; dense layer; output layer
    '''

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=3)
    # cnn_layer1 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(300)
    cnn_dropout1 = Dropout(0.2)
    cnn_dropout2 = Dropout(0.2)
    cnn_batchnormalization = BatchNormalization()
    cnn_repeatvector = RepeatVector(EMBEDDING_DIM)
    cnn_dense1 = Dense(300)
    cnn_timedistributed = TimeDistributed(Dense(1))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    cnn_1 = cnn_layer(embedded_sequences_1)
    # cnn_1 = cnn_layer1(cnn_1)
    cnn_1 = pooling_layer(cnn_1)
    cnn_1 = cnn_dropout1(cnn_1)
    cnn_1 = cnn_dense(cnn_1)
    cnn_1 = cnn_dropout2(cnn_1)
    cnn_1 = cnn_batchnormalization(cnn_1)

    cnn_2 = cnn_layer(embedded_sequences_2) 
    # cnn_2 = cnn_layer1(cnn_2)
    cnn_2 = pooling_layer(cnn_2)
    cnn_2 = cnn_dropout1(cnn_2)
    cnn_2 = cnn_dense(cnn_2)
    cnn_2 = cnn_dropout2(cnn_2)
    cnn_2 = cnn_batchnormalization(cnn_2)
    
    # cnn_1 = cnn_repeatvector(cnn_1)
    # cnn_2 = cnn_repeatvector(cnn_2)

    cnn_1_t = cnn_dense1(cnn_1)
    cnn_2_t = cnn_dense1(cnn_2)

    # cnn_1_t = cnn_timedistributed(cnn_1)
    # cnn_2_t = cnn_timedistributed(cnn_2)

    # cnn_1_t = Permute([2, 1])(cnn_1_t)
    # cnn_2_t = Permute([2, 1])(cnn_2_t)

    a1 = multiply([cnn_1_t, embedded_sequences_1])
    a2 = multiply([cnn_2_t, embedded_sequences_2])
    
    a1 = Permute([2, 1])(a1)
    a2 = Permute([2, 1])(a2)
    
    a1 = Lambda(lambda x: K.sum(x, axis=1))(a1)
    a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)

    a1 = Activation('sigmoid')(a1)
    a2 = Activation('sigmoid')(a2)

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
## CNN based RNN
########################################
def cnn_rnn_add(nb_words, EMBEDDING_DIM, \
                embedding_matrix, MAX_SEQUENCE_LENGTH, \
                num_rnn, num_dense, rate_drop_rnn, \
                rate_drop_dense, act):
    '''
    This is the basic cnn rnn model 

    model: input layer; embedding layer; cnn based attention layer; rnn layer; dense layer; output layer
    '''

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=3)
    # cnn_layer1 = Conv1D(activation="relu", padding="valid", strides=1, filters=64, kernel_size=4)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(300)
    cnn_dropout1 = Dropout(0.2)
    cnn_dropout2 = Dropout(0.2)
    cnn_batchnormalization = BatchNormalization()
    cnn_repeatvector = RepeatVector(EMBEDDING_DIM)
    cnn_dense1 = Dense(300)
    cnn_timedistributed = TimeDistributed(Dense(1))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    cnn_1 = cnn_layer(embedded_sequences_1)
    # cnn_1 = cnn_layer1(cnn_1)
    cnn_1 = pooling_layer(cnn_1)
    cnn_1 = cnn_dropout1(cnn_1)
    cnn_1 = cnn_dense(cnn_1)
    cnn_1 = cnn_dropout2(cnn_1)
    cnn_1 = cnn_batchnormalization(cnn_1)

    cnn_2 = cnn_layer(embedded_sequences_2) 
    # cnn_2 = cnn_layer1(cnn_2)
    cnn_2 = pooling_layer(cnn_2)
    cnn_2 = cnn_dropout1(cnn_2)
    cnn_2 = cnn_dense(cnn_2)
    cnn_2 = cnn_dropout2(cnn_2)
    cnn_2 = cnn_batchnormalization(cnn_2)
    
    # cnn_1 = cnn_repeatvector(cnn_1)
    # cnn_2 = cnn_repeatvector(cnn_2)

    cnn_1_t = cnn_dense1(cnn_1)
    cnn_2_t = cnn_dense1(cnn_2)

    # cnn_1_t = cnn_timedistributed(cnn_1)
    # cnn_2_t = cnn_timedistributed(cnn_2)

    # cnn_1_t = Permute([2, 1])(cnn_1_t)
    # cnn_2_t = Permute([2, 1])(cnn_2_t)

    # a1 = multiply([cnn_1_t, embedded_sequences_1])
    # a2 = multiply([cnn_2_t, embedded_sequences_2])
    
    # a1 = Permute([2, 1])(a1)
    # a2 = Permute([2, 1])(a2)
    
    # a1 = Lambda(lambda x: K.sum(x, axis=1))(a1)
    # a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)

    # a1 = Activation('sigmoid')(a1)
    # a2 = Activation('sigmoid')(a2)

    # embedded_sequences_1 = Permute([2, 1])(embedded_sequences_1)
    # embedded_sequences_2 = Permute([2, 1])(embedded_sequences_2)
    
    x1 = add([cnn_1_t, embedded_sequences_1])
    x2 = add([cnn_2_t, embedded_sequences_2])

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
## CNN based RNN tmp
########################################
def cnn_rnn_tmp(nb_words, EMBEDDING_DIM, \
               embedding_matrix, MAX_SEQUENCE_LENGTH, \
               num_rnn, num_dense, rate_drop_rnn, \
               rate_drop_dense, act):
    '''
    This is the more complex cnn rnn model 

    model: input layer; embedding layer; more complex cnn based attention layer; rnn layer; dense layer; output layer
    '''
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=32, kernel_size=4)
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(300)
    cnn_dropout1 = Dropout(0.2)
    cnn_dropout2 = Dropout(0.2)
    cnn_batchnormalization = BatchNormalization()
    cnn_repeatvector = RepeatVector(EMBEDDING_DIM)
    cnn_dense1 = Dense(300)
    cnn_timedistributed = TimeDistributed(Dense(1))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    conv1a = conv1(embedded_sequences_1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    glob1a = Dropout(0.5)(glob1a)
    glob1a = BatchNormalization()(glob1a)
    conv1b = conv1(embedded_sequences_2)
    glob1b = GlobalAveragePooling1D()(conv1b)
    glob1b = Dropout(0.5)(glob1b)
    glob1b = BatchNormalization()(glob1b)

    conv2a = conv2(embedded_sequences_1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    glob2a = Dropout(0.5)(glob2a)
    glob2a = BatchNormalization()(glob2a)
    conv2b = conv2(embedded_sequences_2)
    glob2b = GlobalAveragePooling1D()(conv2b)
    glob2b = Dropout(0.5)(glob2b)
    glob2b = BatchNormalization()(glob2b)

    conv3a = conv3(embedded_sequences_1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    glob3a = Dropout(0.5)(glob3a)
    glob3a = BatchNormalization()(glob3a)
    conv3b = conv3(embedded_sequences_2)
    glob3b = GlobalAveragePooling1D()(conv3b)
    glob3b = Dropout(0.5)(glob3b)
    glob3b = BatchNormalization()(glob3b)

    conv4a = conv4(embedded_sequences_1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    glob4a = Dropout(0.5)(glob4a)
    glob4a = BatchNormalization()(glob4a)
    conv4b = conv4(embedded_sequences_2)
    glob4b = GlobalAveragePooling1D()(conv4b)
    glob4b = Dropout(0.5)(glob4b)
    glob4b = BatchNormalization()(glob4b)

    conv5a = conv5(embedded_sequences_1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    glob5a = Dropout(0.5)(glob5a)
    glob5a = BatchNormalization()(glob5a)
    conv5b = conv5(embedded_sequences_2)
    glob5b = GlobalAveragePooling1D()(conv5b)
    glob5b = Dropout(0.5)(glob5b)
    glob5b = BatchNormalization()(glob5b)

    conv6a = conv6(embedded_sequences_1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    glob6a = Dropout(0.5)(glob6a)
    glob6a = BatchNormalization()(glob6a)
    conv6b = conv6(embedded_sequences_2)
    glob6b = GlobalAveragePooling1D()(conv6b)
    glob6b = Dropout(0.5)(glob6b)
    glob6b = BatchNormalization()(glob6b)

    cnn_1 = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    cnn_2 = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    cnn_1_t = cnn_dense1(cnn_1)
    cnn_2_t = cnn_dense1(cnn_2)

    a1 = multiply([cnn_1_t, embedded_sequences_1])
    a2 = multiply([cnn_2_t, embedded_sequences_2])
    
    a1 = Permute([2, 1])(a1)
    a2 = Permute([2, 1])(a2)
    
    a1 = Lambda(lambda x: K.sum(x, axis=1))(a1)
    a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)

    a1 = Activation('sigmoid')(a1)
    a2 = Activation('sigmoid')(a2)

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
    '''
    This is the basic baseline model 

    model: input layer; embedding layer; rnn layer; dense layer; output layer
    '''

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    x1 = rnn_layer(embedded_sequences_1)

    x2 = rnn_layer(embedded_sequences_2)

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
## basic cnn
########################################
def basic_cnn(nb_words, EMBEDDING_DIM, \
              embedding_matrix, MAX_SEQUENCE_LENGTH, \
              num_rnn, num_dense, rate_drop_rnn, \
              rate_drop_dense, act):
    '''
    This is the basic cnn model 

    model: input layer; embedding layer; several cnn layer; dense layer; output layer
    '''
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    conv1a = conv1(embedded_sequences_1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(embedded_sequences_2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(embedded_sequences_1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(embedded_sequences_2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(embedded_sequences_1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(embedded_sequences_2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(embedded_sequences_1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(embedded_sequences_2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(embedded_sequences_1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(embedded_sequences_2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(embedded_sequences_1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(embedded_sequences_2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    
    merge = concatenate([diff, mul])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    preds = Dense(1, activation='sigmoid')(x)

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
def basic_attention(nb_words, EMBEDDING_DIM, \
                    embedding_matrix, MAX_SEQUENCE_LENGTH, \
                    num_rnn, num_dense, rate_drop_rnn, \
                    rate_drop_dense, act):
    '''
    This is the basic attention model 

    model: input layer; embedding layer; rnn layer; attention layer; dense layer; output layer
    '''
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    rnn_layer = Bidirectional(GRU(num_rnn, dropout=rate_drop_rnn, recurrent_dropout=rate_drop_rnn, return_sequences=True))
    attention_W = TimeDistributed(Dense(350, activation='tanh'))
    attention_w = TimeDistributed(Dense(1))
    attention_softmax = Activation('softmax')
    attention_sum = Lambda(lambda x: K.sum(x, axis=1))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = rnn_layer(embedded_sequences_1)

    attention1 = attention_W(x1)
    attention1 = attention_w(attention1)
    attention1 = attention_softmax(attention1)
    attention1 = Permute([2, 1])(attention1)
    x1 = Permute([2, 1])(x1)
    x1 = multiply([attention1, x1])
    x1 = Permute([2, 1])(x1)
    x1 = attention_sum(x1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = rnn_layer(embedded_sequences_2)

    attention2 = attention_W(x2)
    attention2 = attention_w(attention2)
    attention2 = attention_softmax(attention2)
    attention2 = Permute([2, 1])(attention2)
    x2 = Permute([2, 1])(x2)
    x2 = multiply([attention2, x2])
    x2 = Permute([2, 1])(x2)
    x2 = attention_sum(x2)

    merged = multiply([x1, x2])
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
