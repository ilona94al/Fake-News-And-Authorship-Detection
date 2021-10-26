import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
import numpy as np


class Task():
    def __init__(self):
        from model.fake_bert_model import FakeBERTModel
        self.model = FakeBERTModel()
        self.max_text_len = self.model.config['max_text_len']

    def get_preprocessed_texts(self, texts):
        preprocessed_texts = []
        for text in texts:
            text_blocks = self.get_preprocessed_text(text)
            preprocessed_texts.extend(block for block in text_blocks)
        return preprocessed_texts

    def get_preprocessed_text(self, text):
        from model.preprocessing import text_preprocessing
        preprocessed_text = text_preprocessing(text)
        from model.preprocessing import separate_text_to_blocks
        text_blocks = separate_text_to_blocks(preprocessed_text, self.max_text_len)
        return text_blocks

    @staticmethod
    def define_expected_classification(real_texts_count, total_texts_count):
        expected_classification = np.empty(total_texts_count, int)
        expected_classification[0:real_texts_count] = 0
        expected_classification[real_texts_count:total_texts_count] = 1
        return expected_classification

    @staticmethod
    def prepare_train_validation_test_sets(texts, y_expected):
        x_train, x, y_train, y = train_test_split(texts, y_expected, train_size=0.7)
        x_test, x_valid, y_test, y_valid = train_test_split(x, y, train_size=0.5)
        return x_train, x_valid, x_test, y_train, y_valid, y_test

    @staticmethod
    def get_categorical_probabilities(y_test, y_train, y_valid):
        y_train_prob = np_utils.to_categorical(y_train)
        y_valid_prob = np_utils.to_categorical(y_valid)
        y_test_prob = np_utils.to_categorical(y_test)
        return y_train_prob, y_valid_prob, y_test_prob

