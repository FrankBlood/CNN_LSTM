# -*- coding:utf8 -*-

'''
The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, recurrent, Convolution1D, GlobalMaxPooling1D
from keras.layers import Dropout, Input, Bidirectional, Merge, RepeatVector, Activation, TimeDistributed, Flatten, RepeatVector, Permute, Lambda
from keras.layers.merge import concatenate, add, dot, multiply
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

########################################
## Bradley  define the model structure
########################################
def bradley(nb_words, EMBEDDING_DIM, \
            embedding_matrix, MAX_SEQUENCE_LENGTH, \
            num_lstm, num_dense, rate_drop_lstm, \
            rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    x1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_1)
    x1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(x1)

    y1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_2)
    y1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(y1)

    merged = concatenate([x1, y1])
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
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
## Abhishek define the model structure
########################################
def abhishek(nb_words, EMBEDDING_DIM, \
             embedding_matrix, MAX_SEQUENCE_LENGTH, \
             num_lstm, num_dense, rate_drop_lstm, \
             rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)

    ########################################
    ## model1
    ########################################
    model1_1 = TimeDistributed(Dense(300, activation='relu'))(embedded_sequences_1)
    model1_1 = Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))(model1_1)
    model1_1 = Dropout(0.2)(model1_1)
    model1_1 = BatchNormalization()(model1_1)

    model1_2 = TimeDistributed(Dense(300, activation='relu'))(embedded_sequences_2)
    model1_2 = Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))(model1_2)
    model1_2 = Dropout(0.2)(model1_2)
    model1_2 = BatchNormalization()(model1_2)
 
    model1 = multiply([model1_1, model1_2])

    ########################################
    ## model2
    ########################################
    model2_1 = Convolution1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(embedded_sequences_1)
    model2_1 = Dropout(0.2)(model2_1)
    model2_1 = Convolution1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(model2_1)
    model2_1 = GlobalMaxPooling1D()(model2_1)
    model2_1 = Dropout(0.2)(model2_1)
    model2_1 = Dense(300)(model2_1)
    model2_1 = Dropout(0.2)(model2_1)
    model2_1 = BatchNormalization()(model2_1)

    model2_2 = Convolution1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(embedded_sequences_2)
    model2_2 = Dropout(0.2)(model2_2)
    model2_2 = Convolution1D(nb_filter=64,
                             filter_length=4,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(model2_2)
    model2_2 = GlobalMaxPooling1D()(model2_2)
    model2_2 = Dropout(0.2)(model2_2)
    model2_2 = Dense(300)(model2_2)
    model2_2 = Dropout(0.2)(model2_2)
    model2_2 = BatchNormalization()(model2_2)

    model2 = multiply([model2_1, model2_2])

    ########################################
    ## model3
    ########################################
    model3_1 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2))(embedded_sequences_1)
    model3_1 = Dropout(0.2)(model3_1)
    model3_1 = BatchNormalization()(model3_1)

    model3_2 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2))(embedded_sequences_2)
    model3_2 = Dropout(0.2)(model3_2)
    model3_2 = BatchNormalization()(model3_2)
    
    model3 = multiply([model3_1, model3_2])

    ########################################
    ## merged model
    ########################################
    
    # merged = concatenate([model1_1, model1_2, model2_1, model2_2, model3_1, model3_2])
    merged = concatenate([model1, model2, model3])
    # merged = concatenate([model2, model3])
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(300)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    # merged = Dense(300)(merged)
    # merged = PReLU()(merged)
    # merged = Dropout(0.2)(merged)
    # merged = BatchNormalization()(merged)

    # merged = Dense(300)(merged)
    # merged = PReLU()(merged)
    # merged = Dropout(0.2)(merged)
    # merged = BatchNormalization()(merged)

    merged = Dense(1)(merged)
    preds = Activation('sigmoid')(merged)

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
## lystdo define the model structure
########################################
def lystdo(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
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
## GRU attention multiply
########################################
def gru_attention_multiply(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = GRU(num_lstm, return_sequences=True, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    attention1 = TimeDistributed(Dense(1, activation='tanh'))(x1)
    attention1 = Flatten()(attention1)
    attention1 = Activation('softmax')(attention1)
    attention1 = RepeatVector(num_lstm)(attention1)
    attention1 = Permute([2, 1])(attention1)
    attention1 = multiply([x1, attention1])
    x1 = Lambda(lambda xin: K.sum(xin, axis=1))(attention1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)
    attention2 = TimeDistributed(Dense(1, activation='tanh'))(y1)
    attention2 = Flatten()(attention2)
    attention2 = Activation('softmax')(attention2)
    attention2 = RepeatVector(num_lstm)(attention2)
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

########################################
## GRU distance
########################################

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def gru_distance(nb_words, EMBEDDING_DIM, \
           embedding_matrix, MAX_SEQUENCE_LENGTH, \
           num_lstm, num_dense, rate_drop_lstm, \
           rate_drop_dense, act):
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = GRU(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    x1 = Dropout(rate_drop_lstm)(x1)
    x1 = BatchNormalization()(x1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)
    y1 = Dropout(rate_drop_lstm)(y1)
    y1 = BatchNormalization()(y1)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x1, y1])

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=distance)
    model.compile(loss=contrastive_loss,
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    # print(STAMP)
    return model

def bilstm_distance_angle_(nb_words, EMBEDDING_DIM, \
                          embedding_matrix, MAX_SEQUENCE_LENGTH, \
                          num_lstm, num_dense, rate_drop_lstm, \
                          rate_drop_dense, act):
    model_q1 = Sequential()
    model_q1.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_q1.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model_q1.add(Bidirectional(LSTM(num_lstm)))
    # model_q1.add(Bidirectional(LSTM(64)))
    model_q1.add(Dropout(rate_drop_lstm))
    model_q1.add(RepeatVector(num_lstm))

    model_q2 = Sequential()
    model_q2.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_q2.add(Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model_q2.add(Bidirectional(LSTM(num_lstm)))
    # model_q2.add(Bidirectional(LSTM(64)))
    model_q2.add(Dropout(rate_drop_lstm))
    model_q2.add(RepeatVector(num_lstm))

    distance = Sequential()
    distance.add(Merge([model_q1, model_q2], mode='sum'))
    angle = Sequential()
    angle.add(Merge([model_q1, model_q2], mode='dot'))

    model = Sequential()
    model.add(Merge([distance, angle], mode='concat'))
    model.add(Bidirectional(LSTM(num_lstm)))
    model.add(Dropout(rate_drop_lstm))
    model.add(Dense(num_dense, activation=act))
    model.add(Dropout(rate_drop_dense))
    model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # adagrad = Adagrad(lr=0.01, epsilon=1e-06)
    # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print(STAMP)
    return model
