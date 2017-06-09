# -*- coding:utf8 -*-

'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7
'''

########################################
## import packages
########################################
import os
import sys
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import *
from string import punctuation
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################
## set directories and parameters
########################################
BASE_DIR = '/home/hegx/Research/Quora_Question_Pairs/data/'

# EMBEDDING_FILE = BASE_DIR + 'vectors_5.bin'
# EMBEDDING_FILE = BASE_DIR + 'vectors_300_8_.bin'
EMBEDDING_FILE = BASE_DIR + 'vectors.bin'
# EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.txt'
# EMBEDDING_FILE = BASE_DIR + 'gensim_vector_window8.bin'
# EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'

TRAIN_DATA_FILE = BASE_DIR + 'train_lower_stemmer.csv'
TEST_DATA_FILE = BASE_DIR + 'test_lower_stemmer.csv'
# TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
# TEST_DATA_FILE = BASE_DIR + 'test.csv'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
# EMBEDDING_DIM = 128
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

num_rnn = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_rnn = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

num_rnn, num_dense = 256, 200
rate_drop_rnn, rate_drop_dense = 0.25, 0.25

act = 'relu'
# re_weight = False # whether to re-weight classes to fit the 17.5% share in test set
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = '_%d_%d_%.2f_%.2f'%(num_rnn, num_dense, rate_drop_rnn, rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False)
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
# print(len(word2vec['a']))
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        # stemmer = SnowballStemmer('english')
        stemmer = PorterStemmer()
        stemmed_words = []
        for word in text:
            try:
                stemmed_words.append(stemmer.stem(word))
            except:
                print word
                stemmed_words.append(word)
        # stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
