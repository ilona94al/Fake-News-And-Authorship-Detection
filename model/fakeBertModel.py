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

class FakeBERTModel():
    def __init__(self ,author_name, folder_path):
        from model.dataManager import DataManager
        self.data_manager=DataManager(author_name,folder_path)
        self.data_manager.data_preprocessing(config)


def build_model():
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer('BERT/preprocessor', name='preprocessing')
    bert_encoder_inputs = preprocessing_layer(input_layer)

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


def fit_model(model, x_train, y_train_prob, x_valid, y_valid_prob, batch_size, epochs):
    # Train the model
    history = model.fit(x_train, y_train_prob, validation_data=(x_valid, y_valid_prob), batch_size=batch_size,
                        epochs=epochs)
    _, accuracy = model.evaluate(x_valid, y_valid_prob, verbose=0)
    print("Accuracy of test groups:", accuracy)
    return model, history


config_128tokens = {
    'max_seq_len': 128,
    'seq_len_cnnb_4': 74,
    'seq_len_cnnb_5': 14
}
config=config_128tokens
