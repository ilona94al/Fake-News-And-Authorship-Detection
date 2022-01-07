from constants import TRAINED_MODELS_PATH
from model.fake_bert_model import FakeBERTModel
import tensorflow as tf

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


def get_distribution(probabilities):
    all = probabilities.shape[0]
    if all == 1:
        real_percent = 100.0 * probabilities[0][0]
        fake_percent = 100.0 * probabilities[0][1]
    else:
        real_percent = 100.0 * sum(probabilities[:, 0]) / all
        fake_percent = 100.0 * sum(probabilities[:, 1]) / all

    return real_percent, fake_percent


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


real_file_path = "DATABASE\\fakenews\db1\\True.csv"
fakes_file_path = "DATABASE\\fakenews\db1\\Fake.csv"
text_col_name = 'text'

#   read tweets
fake_news = read_csv_file_into_array(fakes_file_path, text_col_name)
real_news = read_csv_file_into_array(real_file_path, text_col_name)
fake_news=fake_news[1000:6000]
real_news=real_news[1000:6000]
#   preprocessing
# real_texts = get_preprocessed_texts(real_news)
# fake_texts = get_preprocessed_texts(fake_news)
texts = real_news + fake_news
y_expected = define_expected_classification(len(real_news), len(texts))

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

    chunks = get_preprocessed_text(tweet)
    probabilities = model.predict(tf.constant(chunks))

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
print("acc is: " + str(acc))
print("True predicts: " + str(tp + tn))
print("False predicts: " + str(fp + fn))
