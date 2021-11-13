import os
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
    def __init__(self):
        self.config = config_128tokens

    def build_model(self, num_classes):
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        os.chdir("../BERT/")
        preprocessing_layer = hub.KerasLayer('preprocessor', name='preprocessing')
        bert_encoder_inputs = preprocessing_layer(input_layer)

        bert_encoder = hub.KerasLayer('encoder', trainable=True, name='BERT_encoder')
        bert_outputs = bert_encoder(bert_encoder_inputs)
        embeddings = bert_outputs["sequence_output"]  # [batch_size, seq_length, 768]
        os.chdir("../gui_controllers")
        cnn_parallel_block_1 = tf.keras.layers.Conv1D \
            (filters=128, kernel_size=3, activation='relu', input_shape=(self.config['max_seq_len'], 768))(embeddings)
        cnn_parallel_block_1 = tf.keras.layers.MaxPooling1D \
            (pool_size=5, strides=5)(cnn_parallel_block_1)

        cnn_parallel_block_2 = tf.keras.layers.Conv1D \
            (filters=128, kernel_size=4, activation='relu', input_shape=(self.config['max_seq_len'], 768))(embeddings)
        cnn_parallel_block_2 = tf.keras.layers.MaxPooling1D \
            (pool_size=5, strides=5)(cnn_parallel_block_2)

        cnn_parallel_block_3 = tf.keras.layers.Conv1D \
            (filters=128, kernel_size=5, activation='relu', input_shape=(self.config['max_seq_len'], 768))(embeddings)
        cnn_parallel_block_3 = tf.keras.layers.MaxPooling1D \
            (pool_size=5, strides=5)(cnn_parallel_block_3)

        concatenated_layer = tf.keras.layers.concatenate \
            ([cnn_parallel_block_1, cnn_parallel_block_2, cnn_parallel_block_3], axis=1)

        cnn_block_4 = tf.keras.layers.Conv1D \
            (filters=128, kernel_size=5, activation='relu', input_shape=(self.config['seq_len_cnnb_4'], 128))(
            concatenated_layer)
        cnn_block_4 = tf.keras.layers.MaxPooling1D \
            (pool_size=5, strides=5)(cnn_block_4)

        cnn_block_5 = tf.keras.layers.Conv1D \
            (filters=128, kernel_size=5, activation='relu', input_shape=(self.config['seq_len_cnnb_5'], 128))(
            cnn_block_4)
        cnn_block_5 = tf.keras.layers.MaxPooling1D \
            (pool_size=5, strides=5)(cnn_block_5)

        flatten_layer = tf.keras.layers.Flatten()(cnn_block_5)

        dropped = tf.keras.layers.Dropout(0.2)(flatten_layer)

        dense_layer = tf.keras.layers.Dense \
            (128, activation='relu')(dropped)

        dropped = tf.keras.layers.Dropout(0.2)(dense_layer)
        output_layer = tf.keras.layers.Dense \
            (num_classes, activation='softmax')(dropped)

        self.model = tf.keras.models.Model(input_layer, output_layer)

        self.model.summary()

        # Compile model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                           optimizer=tf.keras.optimizers.Adadelta(0.001))

    def fit_model(self, x_train, y_train_prob, x_valid, y_valid_prob, batch_size, epochs, progress_bar):
        # Train the model
        from model.callback import LoggingCallback
        # self.history = self.model.fit(x=x_train, y=y_train_prob,
        #                               validation_data=(x_valid, y_valid_prob),
        #                               batch_size=batch_size, epochs=epochs,
        #                               callbacks=[LoggingCallback(progress_bar, epochs)])
        self.history = self.model.fit(x=x_train, y=y_train_prob,
                                      validation_data=(x_valid, y_valid_prob),
                                      batch_size=batch_size, epochs=epochs)
        _, self.train_accuracy = self.model.evaluate(x_train, y_train_prob, verbose=0)
        _, self.valid_accuracy = self.model.evaluate(x_valid, y_valid_prob, verbose=0)

    def save_model(self, model_name):
        self.model.save(model_name)

    def test_model(self,x_test,y_test_prob,y_test):
        _, self.test_accuracy = self.model.evaluate(x_test, y_test_prob, verbose=0)
        Y_predicted_prob = self.model.predict(tf.constant(x_test))
        Y_predicted = np.argmax(Y_predicted_prob, -1)
        self.count_well_predicted = np.count_nonzero([y_test == Y_predicted])
        self.count_false_predicted=Y_predicted.shape[0] - self.count_well_predicted
        # print("Accuracy of evaluate new test groups:", self.test_accuracy)
        # print("Number of true predicts:", count_well_predicted)
        # print("Number of false predicts:", Y_predicted.shape[0] - count_well_predicted)
        # -------- Showing results of model training and validation---------------#

        import matplotlib.pyplot as plt
        os.chdir("../PLOTS/")

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.plot(self.test_accuracy)
        plt.title('Model accuracy in epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('ModelAcc.png')

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.plot(self.test_accuracy)
        plt.title('Model loss in epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('ModelLoss.png')

        os.chdir("../gui_controllers")

    def load_model(self, model_path):
        os.chdir("../")
        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        os.chdir("gui_controllers")

    def predict(self, input):
        return self.model.predict(x=input)


config_128tokens = {
    'max_seq_len': 128,
    'seq_len_cnnb_4': 74,
    'seq_len_cnnb_5': 14
}
