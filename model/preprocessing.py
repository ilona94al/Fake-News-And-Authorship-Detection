
import re
from model.fakeBertModel import config
import numpy as np
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


def read_csv_file_into_array(file_name, text_column_name):
    import pandas as pd
    col_list = [text_column_name]
    df = pd.read_csv(file_name, usecols=col_list)
    texts_arr = df[text_column_name].values.tolist()
    return texts_arr


def read_csv_file_into_array(file_name, text_column_name, label_column_name):
    import pandas as pd
    col_list = [text_column_name, label_column_name]
    df = pd.read_csv(file_name, usecols=col_list)
    texts_arr = df[text_column_name].values.tolist()
    labels_arr = df[label_column_name].values.tolist()
    return texts_arr, labels_arr


def separate_text_to_blocks(text):
    words = text.split()
    lines = []
    current = ""
    tokens = 0
    for word in words:
        if tokens > config['max_seq_len'] - 3:
            lines.append(current)
            current = word
            tokens = 0
        else:
            current += word+" "
            tokens += 1
    lines.append(current)
    return lines
