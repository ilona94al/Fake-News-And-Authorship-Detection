import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tweet

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
    arr = []
    i = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            arr[i:] = row[1:2]
            i = i + 1
    return arr


def build_model():
    # https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    # https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3
    # https://www.tensorflow.org/hub/common_saved_model_apis/text
    # https://www.kaggle.com/giovanimachado/hate-speech-bert-cnn-and-bert-mlp-in-tensorflow

    # seq_length = 128 by default
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    bert_encoder_inputs = preprocessing_layer(input_layer)

    # For changing seq_length to maxTextLen...
    # preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3')
    # tokenize = hub.KerasLayer(preprocessor.tokenize)
    # tokenized = tokenize(input_layer)
    # bert_pack_inputs = hub.KerasLayer \
    #     (preprocessor.bert_pack_inputs, arguments=dict(seq_length=maxTextLen))  # Optional argument.

    # bert_encoder_inputs = bert_pack_inputs([tokenized])

    bert_encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
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

    dense_layer = tf.keras.layers.Dense \
        (128, activation='relu')(flatten_layer)

    dropped = tf.keras.layers.Dropout(0.2)(dense_layer)
    output_layer = tf.keras.layers.Dense \
        (num_classes, activation='relu')(dropped)

    model = tf.keras.models.Model(input_layer, output_layer)

    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adadelta(0.001))

    return model


def fit_model(model, x_train, y_train_prob, x_valid, y_valid_prob):
    # Train the model
    history = model.fit(x_train, y_train_prob, validation_data=(x_valid, y_valid_prob), batch_size=10, epochs=5)
    _, accuracy = model.evaluate(x_valid, y_valid_prob, verbose=0)
    print("Accuracy of test groups:", accuracy)
    return model, history


#
# def remove_stop_words(texts):
#     stop_words = set(stopwords.words('english'))
#     new_texts=[]
#     for text in texts:
#         word_tokens = word_tokenize(text)
#         filtered_text = [w for w in word_tokens if not w in stop_words]
#         filtered_text_str=" ".join(word for word in filtered_text)
#         new_texts.append(filtered_text_str)
#
#     return new_texts

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

bert_model_name = 'bert_en_cased_L-12_H-768_A-12'

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}
map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

fake_news = read_file_into_array('DB/fake50.csv')
real_news = read_file_into_array('DB/true50.csv')

news = real_news + fake_news

# define original classification
y_expected = define_expected_classification(len(real_news), len(news))

texts = []
for tweet in news:
    texts.append(text_preprocessing(tweet))

texts_clean = texts

x_train, x, y_train, y = train_test_split(texts_clean, y_expected, train_size=0.7)
x_test, x_valid, y_test, y_valid = train_test_split(x, y, train_size=0.5)

y_train_prob = np_utils.to_categorical(y_train)
y_test_prob = np_utils.to_categorical(y_test)
y_valid_prob = np_utils.to_categorical(y_valid)
num_classes = y_train_prob.shape[1]

model = build_model()
model, history = fit_model(model, tf.constant(x_train), y_train_prob, tf.constant(x_valid), y_valid_prob)

# model.evaluate()

# ------------------------------------
import os
import re

from nltk import word_tokenize
from nltk.corpus import stopwords


def read_books_of_specific_author(author_name):
    res = []

    for book_name in os.listdir(root_books_dir_path + '/' + author_name):
        if book_name.split('.')[1] == 'txt':
            res.append(read_book(author_name, book_name))

    return res


def read_books_of_various_authors(name_to_ignore):
    res = []

    for author_name in os.listdir(root_books_dir_path):
        if author_name != name_to_ignore:
            author_books = read_books_of_specific_author(author_name)
            res.extend(book for book in author_books)

    return res


def read_book(writer_name, book_name):
    with open(root_books_dir_path + '/' + writer_name + '/' + book_name, 'r', encoding='UTF-8') as book_file:
        book_string = book_file.read()
        return book_string


def remove_stop_words(word_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return filtered_text


root_books_dir_path = 'DB/books'

author_books = read_books_of_specific_author(author_name='shakespeare')
different_books = read_books_of_various_authors(name_to_ignore='shakespeare')

books = author_books + different_books

texts = []
for book in books:
    texts.append(text_preprocessing(book))

# define original classification
y_expected = define_expected_classification(len(author_books), len(books))

texts_clean = texts

x_train, x_test, y_train, y_test = train_test_split(texts_clean, y_expected, train_size=0.7)

y_train_prob = np_utils.to_categorical(y_train)
y_test_prob = np_utils.to_categorical(y_test)
num_classes = y_train_prob.shape[1]

model = build_model()
config = config_512tokens
model, history = fit_model(model, tf.constant(x_train), y_train_prob, tf.constant(x_test), y_test_prob)
