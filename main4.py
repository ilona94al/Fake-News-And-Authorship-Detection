import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization  # to create AdamW optimizer
import re

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.utils import np_utils
from nltk import word_tokenize
from nltk.corpus import stopwords


def define_expected_classification(real_texts_count, total_texts_count):
    expected_classification = np.empty(total_texts_count, int)
    expected_classification[0:real_texts_count] = 0
    expected_classification[real_texts_count:total_texts_count] = 1
    return expected_classification


def text_preprocessing(text):
    regex_legal_letters = re.compile('[^ \\t\\n\\v\\f\\ra-zA-Z]')
    text = regex_legal_letters.sub('', text)
    text = text.lower()
    words_arr = text.split()
    clean_word_arr = remove_stop_words(words_arr)
    clean_text = " ".join(word for word in clean_word_arr)
    return clean_text


def remove_stop_words(words_arr):
    stop_words = set(stopwords.words('english'))
    clean_word_arr = [w for w in words_arr if not w in stop_words]
    return clean_word_arr


def read_file_into_array(file_name):
    import csv
    # arr = []
    # i = 0
    import pandas as pd
    col_list = ["text"]
    df = pd.read_csv(file_name, usecols=col_list)
    arr = df['text'].values.tolist()

    # with open(file_name, 'r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         arr[i:] = row[1:2]
    #         i = i + 1

    return arr


def seperate_to_blocks(book):
    words = iter(book.split())
    lines, current = [], next(words)
    tokens = 0
    for word in words:
        if tokens > config['max_seq_len'] - 3:
            lines.append(current)
            current = word
            tokens = 0
        else:
            current += " " + word
            tokens += 1
    lines.append(current)
    return lines


# def build_model():
#     # https://www.tensorflow.org/text/tutorials/classify_text_with_bert
#     # https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3
#     # https://www.tensorflow.org/hub/common_saved_model_apis/text
#     # https://www.kaggle.com/giovanimachado/hate-speech-bert-cnn-and-bert-mlp-in-tensorflow
#
#     # seq_length = 128 by default
#
#     input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     preprocessing_layer = hub.KerasLayer('BERT/preprocessor', name='preprocessing')
#     bert_encoder_inputs = preprocessing_layer(input_layer)
#
#     # For changing seq_length to maxTextLen...
#     # preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3')
#     # tokenize = hub.KerasLayer(preprocessor.tokenize)
#     # tokenized = tokenize(input_layer)
#     # bert_pack_inputs = hub.KerasLayer \
#     #     (preprocessor.bert_pack_inputs, arguments=dict(seq_length=maxTextLen))  # Optional argument.
#
#     # bert_encoder_inputs = bert_pack_inputs([tokenized])
#
#     bert_encoder = hub.KerasLayer('BERT/encoder', trainable=True, name='BERT_encoder')
#     bert_outputs = bert_encoder(bert_encoder_inputs)
#     embeddings = bert_outputs["sequence_output"]  # [batch_size, seq_length, 768]
#
#     cnn_parallel_block_1 = tf.keras.layers.Conv1D \
#         (filters=128, kernel_size=3, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
#     cnn_parallel_block_1 = tf.keras.layers.MaxPooling1D \
#         (pool_size=5, strides=5)(cnn_parallel_block_1)
#
#     cnn_parallel_block_2 = tf.keras.layers.Conv1D \
#         (filters=128, kernel_size=4, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
#     cnn_parallel_block_2 = tf.keras.layers.MaxPooling1D \
#         (pool_size=5, strides=5)(cnn_parallel_block_2)
#
#     cnn_parallel_block_3 = tf.keras.layers.Conv1D \
#         (filters=128, kernel_size=5, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
#     cnn_parallel_block_3 = tf.keras.layers.MaxPooling1D \
#         (pool_size=5, strides=5)(cnn_parallel_block_3)
#
#     concatenated_layer = tf.keras.layers.concatenate \
#         ([cnn_parallel_block_1, cnn_parallel_block_2, cnn_parallel_block_3], axis=1)
#
#     cnn_block_4 = tf.keras.layers.Conv1D \
#         (filters=128, kernel_size=5, activation='relu', input_shape=(config['seq_len_cnnb_4'], 128))(concatenated_layer)
#     cnn_block_4 = tf.keras.layers.MaxPooling1D \
#         (pool_size=5, strides=5)(cnn_block_4)
#
#     cnn_block_5 = tf.keras.layers.Conv1D \
#         (filters=128, kernel_size=5, activation='relu', input_shape=(config['seq_len_cnnb_5'], 128))(cnn_block_4)
#     cnn_block_5 = tf.keras.layers.MaxPooling1D \
#         (pool_size=5, strides=5)(cnn_block_5)
#
#     flatten_layer = tf.keras.layers.Flatten()(cnn_block_5)
#
#     dropped = tf.keras.layers.Dropout(0.2)(flatten_layer)
#
#     dense_layer = tf.keras.layers.Dense \
#         (128, activation='relu')(dropped)
#
#     dropped = tf.keras.layers.Dropout(0.2)(dense_layer)
#     output_layer = tf.keras.layers.Dense \
#         (num_classes, activation='softmax')(dropped)
#
#     model = tf.keras.models.Model(input_layer, output_layer)
#
#     model.summary()
#
#     # Compile model
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta(0.001))
#
#     return model

def build_model():
    # https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    # https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3
    # https://www.tensorflow.org/hub/common_saved_model_apis/text
    # https://www.kaggle.com/giovanimachado/hate-speech-bert-cnn-and-bert-mlp-in-tensorflow

    # seq_length = 128 by default

    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer('BERT/preprocessor', name='preprocessing')
    bert_encoder_inputs = preprocessing_layer(input_layer)

    # For changing seq_length to maxTextLen...
    # preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3')
    # tokenize = hub.KerasLayer(preprocessor.tokenize)
    # tokenized = tokenize(input_layer)
    # bert_pack_inputs = hub.KerasLayer \
    #     (preprocessor.bert_pack_inputs, arguments=dict(seq_length=maxTextLen))  # Optional argument.

    # bert_encoder_inputs = bert_pack_inputs([tokenized])

    bert_encoder = hub.KerasLayer('BERT/encoder', trainable=True, name='BERT_encoder')
    bert_outputs = bert_encoder(bert_encoder_inputs)
    embeddings = bert_outputs["sequence_output"]  # [batch_size, seq_length, 768]

    cnn_parallel_block_1 = tf.keras.layers.Conv1D \
        (filters=128, kernel_size=3, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
    cnn_parallel_block_1 = tf.keras.layers.MaxPooling1D \
        (pool_size=5, strides=5)(cnn_parallel_block_1)

    cnn_parallel_block_2 = tf.keras.layers.Conv1D \
        (filters=128, kernel_size=4, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
    cnn_parallel_block_2 = tf.keras.layers.MaxPooling1D \
        (pool_size=5, strides=5)(cnn_parallel_block_2)

    cnn_parallel_block_3 = tf.keras.layers.Conv1D \
        (filters=128, kernel_size=5, activation='relu', input_shape=(config['max_seq_len'], 768))(embeddings)
    cnn_parallel_block_3 = tf.keras.layers.MaxPooling1D \
        (pool_size=5, strides=5)(cnn_parallel_block_3)

    concatenated_layer = tf.keras.layers.concatenate \
        ([cnn_parallel_block_1, cnn_parallel_block_2, cnn_parallel_block_3], axis=1)

    cnn_block_4 = tf.keras.layers.Conv1D \
        (filters=128, kernel_size=5, activation='relu', input_shape=(config['seq_len_cnnb_4'], 128))(concatenated_layer)
    cnn_block_4 = tf.keras.layers.MaxPooling1D \
        (pool_size=5, strides=5)(cnn_block_4)

    cnn_block_5 = tf.keras.layers.Conv1D \
        (filters=128, kernel_size=5, activation='relu', input_shape=(config['seq_len_cnnb_5'], 128))(cnn_block_4)
    cnn_block_5 = tf.keras.layers.MaxPooling1D \
        (pool_size=5, strides=5)(cnn_block_5)

    flatten_layer = tf.keras.layers.Flatten()(cnn_block_5)

    dropped = tf.keras.layers.Dropout(0.2)(flatten_layer)

    dense_layer = tf.keras.layers.Dense \
        (128, activation='relu')(dropped)

    dropped = tf.keras.layers.Dropout(0.2)(dense_layer)
    output_layer = tf.keras.layers.Dense \
        (num_classes, activation='softmax')(dropped)

    model = tf.keras.models.Model(input_layer, output_layer)

    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta(0.001))

    return model
def fit_model(model, x_train, y_train_prob, x_valid, y_valid_prob):
    # Train the model
    history = model.fit(x_train, y_train_prob, validation_data=(x_valid, y_valid_prob), batch_size=10, epochs=3)
    _, accuracy = model.evaluate(x_valid, y_valid_prob, verbose=0)
    print("Accuracy of test groups:", accuracy)
    return model, history


config_128tokens = {
    'max_seq_len': 128,
    'seq_len_cnnb_4': 74,
    'seq_len_cnnb_5': 14
}

config_512tokens = {
    'max_seq_len': 512,
    'seq_len_cnnb_4': 304,
    'seq_len_cnnb_5': 60
}

config = config_128tokens

fake_news = read_file_into_array('DATABASE/fakenews/db1/fake50.csv')
real_news = read_file_into_array('DATABASE/fakenews/db1/true50.csv')

real_texts = []
for tweet in real_news:
    preprocessed_text = text_preprocessing(tweet)
    tweet_blocks = seperate_to_blocks(preprocessed_text)
    real_texts.extend((block for block in tweet_blocks))

fake_texts = []
for tweet in fake_news:
    preprocessed_text = text_preprocessing(tweet)
    tweet_blocks = seperate_to_blocks(preprocessed_text)
    fake_texts.extend((block for block in tweet_blocks))

texts = real_texts + fake_texts

# define original classification
y_expected = define_expected_classification(len(real_texts), len(texts))

x_train, x, y_train, y = train_test_split(texts, y_expected, train_size=0.1)
x_test, x_valid, y_test, y_valid = train_test_split(x, y, train_size=0.5)

y_train_prob = np_utils.to_categorical(y_train)
y_test_prob = np_utils.to_categorical(y_test)
y_valid_prob = np_utils.to_categorical(y_valid)
num_classes = y_train_prob.shape[1]

new_model = True
model_name = 'FakeBERTModel.h5'
trained_model_name = 'TrainedFakeBERTModel.h5'
if new_model == True:
    model = build_model()
    model.save(model_name)
    model, history = fit_model(model, tf.constant(x_train), y_train_prob, tf.constant(x_valid), y_valid_prob)
    model.save(trained_model_name)
else:
    model = tf.keras.models.load_model(trained_model_name, custom_objects={'KerasLayer': hub.KerasLayer})

_, accuracy = model.evaluate(tf.constant(x_test), y_test_prob, verbose=0)
# print("Accuracy of evaluate new test groups:", accuracy)
#
# Y_predicted_prob = model.predict(tf.constant(x_test))
# Y_predicted = np.argmax(Y_predicted_prob, -1)
# count_well_predicted = np.count_nonzero([y_test == Y_predicted])
#
# print("Number of true predicts:", count_well_predicted)
# print("Number of false predicts:", Y_predicted.shape[0] - count_well_predicted)

#-------- Showing results of model training and validation---------------#

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(accuracy)
plt.title('Model accuracy in epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig('ModelAcc.png')
plt.savefig('plots/ModelAcc.png')
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(accuracy)
# plt.title('Model loss in epoch')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
