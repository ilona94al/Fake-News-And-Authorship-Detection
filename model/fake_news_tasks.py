from constants import PLOTS_PATH, TRAINED_MODELS_PATH
from model.task import Task
import numpy as np


class FakeNewsTask1(Task):
    def __init__(self, mixed_file_path, text_col_name, label_col_name, real_label_val, fakes_label_val, batch_size,
                 epochs):
        super().__init__(batch_size, epochs)
        self.error = False
        try:
            #   read tweets
            news, labels = self.read_csv_file_into_array(mixed_file_path, text_col_name, label_col_name)

            #   preprocessing
            texts, clean_labels = self.get_preprocessed_texts_for_one_file(news, labels, real_label_val,
                                                                           fakes_label_val)
        except:
            self.error = True
            return

        #   set original classification
        y_expected = self.get_classification_in_appropriate_format(clean_labels)

        #   split all data to train set, validation set and test set
        self.prepare_train_validation_test_sets(texts, y_expected)

        #   set probabilities for each text to belong for each label
        self.get_categorical_probabilities(self.y_test, self.y_train, self.y_valid)

        self.model_path = TRAINED_MODELS_PATH + "FakeNews/"

    #   gets file name, the name for a column with tweet content, and name for a column with label
    #   returns 2 arrays: 1st with tweets, 2nd with labels (corresponding to tweets by order)
    @staticmethod
    def read_csv_file_into_array(file_name, text_column_name, label_column_name):
        import pandas as pd
        col_list = [text_column_name, label_column_name]
        df = pd.read_csv(file_name, usecols=col_list)
        texts_arr = df[text_column_name].values.tolist()
        labels_arr = df[label_column_name].values.tolist()
        return texts_arr, labels_arr

    #   gets texts array, an array with corresponding labels, values for real and fake label
    #   returns 2 arrays: 1st with preprocessed texts, 2nd with corresponding labels
    def get_preprocessed_texts_for_one_file(self, texts, labels, real_label_val, fakes_label_val):
        clean_labels = []
        clean_texts = []
        j = 0

        label_is_integer = self.is_numeric(fakes_label_val) and self.is_numeric(real_label_val)
        for i, tweet in enumerate(texts):
            if isinstance(tweet, str):
                if label_is_integer:
                    if self.is_numeric(str(labels[i])) and self.is_one_of_labels(int(labels[i]), int(real_label_val),
                                                                                 int(fakes_label_val)):
                        j = self.update(clean_labels, clean_texts, j, labels[i], tweet)
                elif self.is_one_of_labels(labels[i], real_label_val, fakes_label_val):
                    j = self.update(clean_labels, clean_texts, j, labels[i], tweet)

        return clean_texts, clean_labels

    def update(self, clean_labels, clean_texts, j, label, tweet):
        tweet_blocks = self.get_preprocessed_text("" + tweet)
        clean_texts.extend(block for block in tweet_blocks)
        clean_labels.extend([label for x in range(j, j + len(tweet_blocks))])
        j = j + len(tweet_blocks)
        return j

    #   get array with labels
    #   returns array with same labels in appropriate format
    @staticmethod
    def get_classification_in_appropriate_format(clean_labels):
        y_expected = np.empty(len(clean_labels), int)
        y_expected[0:len(clean_labels)] = clean_labels[0:len(clean_labels)]
        return y_expected

    @staticmethod
    def is_numeric(num_str):
        return num_str.isnumeric() or num_str.replace('.', '').isdigit()

    @staticmethod
    def is_one_of_labels(label, real_label_val, fakes_label_val):
        return label == fakes_label_val or label == real_label_val


class FakeNewsTask2(Task):
    def __init__(self, real_file_path, fakes_file_path, text_col_name, batch_size, epochs):
        super().__init__(batch_size, epochs)
        self.error = False
        try:
            #   read tweets
            fake_news = self.read_csv_file_into_array(fakes_file_path, text_col_name)
            real_news = self.read_csv_file_into_array(real_file_path, text_col_name)

            #   preprocessing
            real_texts = self.get_preprocessed_texts(real_news)
            fake_texts = self.get_preprocessed_texts(fake_news)
        except:
            self.error = True
            return

        #   Union
        texts = real_texts + fake_texts

        #   set original classification
        y_expected = self.define_expected_classification(len(real_texts), len(texts))

        #   split all data to train set, validation set and test set
        self.prepare_train_validation_test_sets(texts, y_expected)

        #   set probabilities for each text to belong for each label
        self.get_categorical_probabilities(self.y_test, self.y_train, self.y_valid)

        self.model_path = TRAINED_MODELS_PATH + "FakeNews/"

    #   gets file name,abd  the name for a column with tweet content
    #   returns array with tweets
    @staticmethod
    def read_csv_file_into_array(file_name, text_column_name):
        import pandas as pd
        col_list = [text_column_name]
        df = pd.read_csv(file_name, usecols=col_list)
        texts_arr = df[text_column_name].values.tolist()
        return texts_arr
