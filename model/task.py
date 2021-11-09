import os
import threading

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
import numpy as np
import tensorflow as tf
from multiprocessing import Process

class Task():
    def __init__(self, batch_size, epochs):
        from model.fake_bert_model import FakeBERTModel
        self.model = FakeBERTModel()
        self.max_text_len = self.model.config['max_seq_len']
        self.batch_size, self.epochs = batch_size, epochs

    #   gets array with texts
    #   returns an array with preprocessed texts
    def get_preprocessed_texts(self, texts):
        preprocessed_texts = []
        for text in texts:
            text_blocks = self.get_preprocessed_text(text)
            preprocessed_texts.extend(block for block in text_blocks)
        return preprocessed_texts

    #   gets text
    #   returns preprocessed text
    def get_preprocessed_text(self, text):
        from model.preprocessing import text_preprocessing
        preprocessed_text = text_preprocessing(text)
        from model.preprocessing import separate_text_to_blocks
        text_blocks = separate_text_to_blocks(preprocessed_text, self.max_text_len)
        return text_blocks

    #   gets texts and corresponding labels
    #   split data to train set, validation set and test set
    def prepare_train_validation_test_sets(self, texts, y_expected):
        self.x_train, x, self.y_train, y = train_test_split(texts, y_expected, train_size=0.7)
        self.x_test, self.x_valid, self.y_test, self.y_valid = train_test_split(x, y, train_size=0.5)

    #   gets 3 arrays with labels
    #   sets probabilities for each text to belong for each label
    #   ----    Real    |   Fake
    #   text1   0.0     |   1.0
    #   text2   1.0     |   0.0
    #   ....    ...         ...
    def get_categorical_probabilities(self, y_test, y_train, y_valid):
        self.y_train_prob = np_utils.to_categorical(y_train)
        self.y_valid_prob = np_utils.to_categorical(y_valid)
        self.y_test_prob = np_utils.to_categorical(y_test)
        self.num_classes = self.y_train_prob.shape[1]

    def run_train_model(self):
        self.model.build_model(self.num_classes)
        self.model.fit_model \
            (x_train=tf.constant(self.x_train), y_train_prob=self.y_train_prob,
             x_valid=tf.constant(self.x_valid), y_valid_prob=self.y_valid_prob,
             batch_size=self.batch_size, epochs=self.epochs, progress_bar=self.progress_bar)
        self.running = False

    def start_train_model(self, progress_bar):
        self.running = True
        self.progress_bar = progress_bar
        self.current_epoch = 0
        self.t = threading.Thread(target=self.run_train_model)
        self.t.start()


    def save_model(self, dir_name,model_name):

        os.chdir("../Model1/"+dir_name)
        self.model.save_model(model_name)
        os.chdir("../../gui")
        # todo: show error popup if name exists

    def test_model(self):
        self.model.test_model(x_test=tf.constant(self.x_test), y_test_prob=self.y_test_prob, y_test=self.y_test)

    # get a number of real texts and number of all texts in the dataset
    # returns labels array with label 0 for real texts, and 1 for fake texts.

    @staticmethod
    def define_expected_classification(real_texts_count, total_texts_count):
        expected_classification = np.empty(total_texts_count, int)
        expected_classification[0:real_texts_count] = 0
        expected_classification[real_texts_count:total_texts_count] = 1
        return expected_classification
