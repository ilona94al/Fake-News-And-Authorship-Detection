from constants import TRAINED_MODELS_PATH
from model.fake_bert_model import FakeBERTModel
from model.fake_news_detection import FakeNewsDetection
from run_books_detection import get_distribution
import tensorflow as tf

import numpy as np


def get_preprocessed_text(text):
    from model.preprocessing import text_preprocessing
    preprocessed_text = text_preprocessing(text)
    from model.preprocessing import separate_text_to_blocks
    text_blocks = separate_text_to_blocks(preprocessed_text, 128)
    return text_blocks


def read_csv_file_into_array(file_name, text_column_name, label_column_name):
    import pandas as pd
    col_list = [text_column_name, label_column_name]
    df = pd.read_csv(file_name, usecols=col_list)
    texts_arr = df[text_column_name].values.tolist()
    labels_arr = df[label_column_name].values.tolist()
    return texts_arr, labels_arr


def get_preprocessed_texts_for_one_file(texts, labels, real_label_val, fakes_label_val):
    clean_labels = []
    clean_texts = []
    j = 0

    label_is_integer = is_numeric(fakes_label_val) and is_numeric(real_label_val)
    for i, tweet in enumerate(texts):
        if isinstance(tweet, str):
            if label_is_integer:
                if is_numeric(str(labels[i])) and is_one_of_labels(int(labels[i]), int(real_label_val),
                                                                   int(fakes_label_val)):
                    j = update(clean_labels, clean_texts, j, labels[i], tweet)
            elif is_one_of_labels(labels[i], real_label_val, fakes_label_val):
                j = update(clean_labels, clean_texts, j, labels[i], tweet)

    return clean_texts, clean_labels


def is_numeric(num_str):
    return num_str.isnumeric() or num_str.replace('.', '').isdigit()


def is_one_of_labels(label, real_label_val, fakes_label_val):
    return label == fakes_label_val or label == real_label_val


def update(clean_labels, clean_texts, j, label, tweet):
    tweet_blocks = get_preprocessed_text("" + tweet)
    clean_texts.extend(block for block in tweet_blocks)
    clean_labels.extend([label for x in range(j, j + len(tweet_blocks))])
    j = j + len(tweet_blocks)
    return j


def write_results_to_file(file_path, text, real_percent, real_class):
    import csv
    from pathlib import Path

    # my_file = Path("../"+file_path)
    my_file = Path(file_path)

    file_exist = my_file.exists()

    #  with open("../"+file_path, 'a', encoding='UTF8') as f:
    with open(file_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        if not file_exist:
            header = ['text', 'reliable news percent', 'real classification (if exist)']
            writer.writerow(header)
        data = [text, "{:.1f}%".format(real_percent), real_class]
        writer.writerow(data)


mixed_file_path = "DATABASE\\fakenews\db2\\test.csv"
text_col_name = 'text'
label_col_name = 'label'
real_label_val_ = '0'
fakes_label_val_ = '1'
#   read tweets
news_, labels_ = read_csv_file_into_array(mixed_file_path, text_col_name, label_col_name)

#   preprocessing
texts_, clean_labels = get_preprocessed_texts_for_one_file(news_, labels_, real_label_val_, fakes_label_val_)
tp = 0
tn = 0
fn = 0
fp = 0
model_name = "Fake_News_1"
model = FakeBERTModel()
#   loads the relevant model according to name and type
model.load_model(TRAINED_MODELS_PATH + "FakeNews" + "/" + model_name + ".h5")
results_csv_path = model_name + "-db2- Detection results.csv"
for i, tweet in enumerate(texts_):

    probabilities = model.predict(tf.constant(tweet))

    real_percent, fake_percent = get_distribution(probabilities)

    write_results_to_file(model_name, tweet, real_percent,
                          clean_labels[i])
    if real_percent > fake_percent:
        if int(clean_labels[i]) == int(real_label_val_):  # real
            tp = tp + 1
        else:  # actually fake
            fp = fp + 1
    elif int(clean_labels[i]) == int(fakes_label_val_):  # fake
        tn = tn + 1
    else:  # actually true
        fn = fn + 1

acc = (tp + tn) / (tp + tn + fp + fn)
print("acc is: " + acc)
print("True predicts: " + tp + tn)
print("False predicts: " + fp + fn)
