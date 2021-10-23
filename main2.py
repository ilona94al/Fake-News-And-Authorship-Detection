import os
import shutil


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

import re

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils


def define_expected_classification(fake_num, all_n):
    expected_classification = np.empty(all_n, int)
    expected_classification[0:fake_num] = 0
    expected_classification[fake_num:all_n] = 1
    return expected_classification


# cleaner of the text
def preProcess(string):
    string = re.sub(" +", " ", string)
    string = re.sub("'", "", string)
    string = re.sub("[^a-zA-Z ]", " ", string)
    string = string.lower()
    return string


def read_file_into_array(file_name):
    import csv
    arr = []
    i = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            arr[i:] = row[1:2]
            i = i + 1
    return arr


# tf.get_logger().setLevel('ERROR')
#
# bert_model_name = 'bert_en_cased_L-12_H-768_A-12'
#
# map_name_to_handle = {
#     'bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
#     'bert_en_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
#     'bert_multi_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
#     'small_bert/bert_en_uncased_L-2_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-2_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-2_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-2_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-4_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-4_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-4_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-4_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-6_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-6_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-6_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-6_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-8_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-8_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-8_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-8_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-10_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-10_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-10_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-10_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-12_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-12_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-12_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
#     'albert_en_base':
#         'https://tfhub.dev/tensorflow/albert_en_base/2',
#     'electra_small':
#         'https://tfhub.dev/google/electra_small/2',
#     'electra_base':
#         'https://tfhub.dev/google/electra_base/2',
#     'experts_pubmed':
#         'https://tfhub.dev/google/experts/bert/pubmed/2',
#     'experts_wiki_books':
#         'https://tfhub.dev/google/experts/bert/wiki_books/2',
#     'talking-heads_base':
#         'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
# }
# map_model_to_preprocess = {
#     'bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'bert_en_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'bert_multi_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
#     'albert_en_base':
#         'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
#     'electra_small':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'electra_base':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'experts_pubmed':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'experts_wiki_books':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'talking-heads_base':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
# }
#
# tfhub_handle_encoder = map_name_to_handle[bert_model_name]
# tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
#
# print(f'BERT model selected           : {tfhub_handle_encoder}')
# print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
#
# bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
#
fake_news = read_file_into_array('database/fakenews/db1/fake50.csv')
real_news = read_file_into_array('database/fakenews/db1/true50.csv')

news = fake_news + real_news

# define original classification
y_expected = define_expected_classification(len(fake_news), len(news))

texts = []
for text in news:
    texts.append(preProcess(text))

texts_clean = texts
#
# # todo: delete Tokenizer and pad_sequnces and replace it with bert preproccess (also need to change 128 tokens to 1000 tokens)
#
# preprocessed = bert_preprocess_model(texts)
#
# # text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# # preprocessor = hub.KerasLayer(
# #     "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3")
# # encoder_inputs = preprocessor(text_input)
#
# preprocessor = hub.load(
#     "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3")
# # Tokenize batches of both text inputs.
# text_premises = tf.constant(texts)
# tokenize = hub.KerasLayer(preprocessor.tokenize)
# tokenized_premises = tokenize(text_premises)
#
# bert_pack_inputs = hub.KerasLayer(
#     preprocessor.bert_pack_inputs,
#     arguments=dict(seq_length=128))  # Optional argument.
# encoder_inputs = bert_pack_inputs([tokenized_premises])
#
# bert_model = hub.KerasLayer(tfhub_handle_encoder)
# bert_results = bert_model(encoder_inputs)
#
# preprocessor = hub.load(
#     "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3")
#
# # Step 1: tokenize batches of text inputs.
# text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string), texts]  # This SavedModel accepts up to 2 text inputs.
# tokenize = hub.KerasLayer(preprocessor.tokenize)
# tokenized_inputs = [tokenize(segment) for segment in text_inputs]

# Step 2 (optional): modify tokenized inputs.
pass

# Step 3: pack input sequences for the Transformer encoder.
# seq_length = 128  # Your choice here.
# bert_pack_inputs = hub.KerasLayer(
#     preprocessor.bert_pack_inputs,
#     arguments=dict(seq_length=seq_length))  # Optional argument.
# encoder_inputs = bert_pack_inputs(tokenized_inputs)

#bert_model = hub.KerasLayer(tfhub_handle_encoder)
#bert_results = bert_model(encoder_inputs)
# vocab_size = ?
# maxTextLen = x['input_word_ids'].shape[1] #128 now. we want 1000 -> how to do this ?

tokenizer = Tokenizer(filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                      split=' ', char_level=False)
tokenizer.fit_on_texts(texts_clean)
seq = tokenizer.texts_to_sequences(texts_clean)
maxTextLen = 1000
x = pad_sequences(seq, maxlen=maxTextLen)

vocab_size = x.max()

x_train, x_test, y_train, y_test = train_test_split(x, y_expected, train_size=0.7)

# todo: end

y_train_prob = np_utils.to_categorical(y_train)
y_test_prob = np_utils.to_categorical(y_test)
num_classes = y_train_prob.shape[1]


def build_model():
    import tensorflow as tf

    # todo: delete Embedding and BLSTM and replace it with bert encoder...
    #
    # bert_model = hub.KerasLayer(tfhub_handle_encoder)
    #
    # bert_results = bert_model(text_preprocessed)
    #
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    # net = outputs['pooled_output'] # [batch_size, 768].
    net = sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768]

    input_shape = tf.keras.layers.Input(shape=(maxTextLen))

    emb = tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=maxTextLen)(input_shape)

    blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.5),
                                          input_shape=(maxTextLen, 128))(emb)
    # todo: end

    cnn_parallel_block_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                                                  input_shape=(maxTextLen, 100))(
        blstm)

    cnn_parallel_block_1 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(cnn_parallel_block_1)
    cnn_parallel_block_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu',
                                                  input_shape=(maxTextLen, 100))(
        blstm)
    cnn_parallel_block_2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(cnn_parallel_block_2)
    cnn_parallel_block_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu',
                                                  input_shape=(maxTextLen, 100))(
        blstm)
    cnn_parallel_block_3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(cnn_parallel_block_3)

    concatenated_layer = tf.keras.layers.concatenate([cnn_parallel_block_1, cnn_parallel_block_2, cnn_parallel_block_3],
                                                     axis=1)
    cnn_block_4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(597, 128))(
        concatenated_layer)
    cnn_block_4 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=5)(cnn_block_4)
    cnn_block_5 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(118, 128))(
        cnn_block_4)
    cnn_block_5 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=38)(cnn_block_5)

    flatten_layer = tf.keras.layers.Flatten()(cnn_block_5)

    # model.add(Dense(8, activation='relu'))

    dense_layer = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='relu')(dense_layer)

    model = tf.keras.models.Model(input_shape, output_layer)
    model.summary()
    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta(0.001))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('model.h5')
    return model


def fit_model(model, x_train, y_train_prob, x_test, y_test_prob):
    # Train the model
    history = model.fit(x_train, y_train_prob, validation_data=(x_test, y_test_prob), batch_size=10, epochs=10)
    _, accuracy = model.evaluate(x_test, y_test_prob, verbose=0)
    print("Accuracy of test groups:", accuracy)
    return model, history


model = build_model()
model, history = fit_model(model, x_train, y_train_prob, x_test, y_test_prob)
