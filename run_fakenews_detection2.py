from constants import TRAINED_MODELS_PATH
from model.fake_bert_model import FakeBERTModel
from model.fake_news_detection import FakeNewsDetection
from run_books_detection import get_distribution
import tensorflow as tf

import numpy as np

from run_fakenews_detection import write_results_to_file


def read_csv_file_into_array(file_name, text_column_name):
    import pandas as pd
    col_list = [text_column_name]
    df = pd.read_csv(file_name, usecols=col_list)
    texts_arr = df[text_column_name].values.tolist()
    return texts_arr


def get_preprocessed_texts(texts):
    preprocessed_texts = []
    for text in texts:
        text_blocks = get_preprocessed_text(text)
        preprocessed_texts.extend(block for block in text_blocks)
    return preprocessed_texts


def get_preprocessed_text(text):
    from model.preprocessing import text_preprocessing
    preprocessed_text = text_preprocessing(text)
    from model.preprocessing import separate_text_to_blocks
    text_blocks = separate_text_to_blocks(preprocessed_text, 128)
    return text_blocks


def define_expected_classification(real_texts_count, total_texts_count):
    import numpy as np
    expected_classification = np.empty(total_texts_count, int)
    expected_classification[0:real_texts_count] = 0
    expected_classification[real_texts_count:total_texts_count] = 1
    return expected_classification


real_file_path = "DATABASE\\fakenews\db1\\real_test2.csv"
fakes_file_path = "DATABASE\\fakenews\db1\\fake_test2.csv"
text_col_name = 'text'

#   read tweets
fake_news = read_csv_file_into_array(fakes_file_path, text_col_name)
real_news = read_csv_file_into_array(real_file_path, text_col_name)

#   preprocessing
real_texts = get_preprocessed_texts(real_news)
fake_texts = get_preprocessed_texts(fake_news)
texts = real_texts + fake_texts
y_expected = define_expected_classification(len(real_texts), len(texts))

tp = 0
tn = 0
fn = 0
fp = 0
model_name = "Fake_News_1"
model = FakeBERTModel()
#   loads the relevant model according to name and type
model.load_model(TRAINED_MODELS_PATH + "FakeNews" + "/" + model_name + ".h5")
results_csv_path = model_name + "-db1- Detection results.csv"
for i, tweet in enumerate(texts):
    probabilities = model.predict(tf.constant(tweet))

    real_percent, fake_percent = get_distribution(probabilities)

    write_results_to_file(results_csv_path, tweet, real_percent,
                                    y_expected[i])
    if real_percent > fake_percent:
        if int(y_expected[i]) == 0:  # real
            tp = tp + 1
        else:  # actually fake
            fp = fp + 1
    elif int(y_expected[i]) == 1:  # fake
        tn = tn + 1
    else:  # actually true
        fn = fn + 1

acc = (tp + tn) / (tp + tn + fp + fn)
print("acc is: " + acc)
print("True predicts: " + tp + tn)
print("False predicts: " + fp + fn)
