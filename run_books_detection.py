import os

from constants import TRAINED_MODELS_PATH
from model.fake_bert_model import FakeBERTModel
import tensorflow as tf

import numpy as np


def read_book(book_dir_path):
    with open(book_dir_path, 'r', encoding='UTF-8') as book_file:
        book_string = book_file.read()
        return book_string


def get_preprocessed_text(input, max_text_len):
    from model.preprocessing import text_preprocessing
    preprocessed_input = text_preprocessing(input)
    from model.preprocessing import separate_text_to_blocks
    input_blocks = separate_text_to_blocks(preprocessed_input, max_text_len)
    return input_blocks


def get_distribution(probabilities):
    all = probabilities.shape[0]
    if all == 1:
        real_percent = 100.0 * probabilities[0][0]
        fake_percent = 100.0 * probabilities[0][1]
    else:
        real_percent = 100.0 * sum(probabilities[:, 0]) / all
        fake_percent = 100.0 * sum(probabilities[:, 1]) / all

    return real_percent, fake_percent


def write_results_to_file(file_path, author_name, book_name, percent):
    import csv
    from pathlib import Path

    my_file = Path(file_path)

    file_exist = my_file.exists()

    with open(file_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        if not file_exist:
            header = ['author name', 'book name', 'percent that written by author']
            writer.writerow(header)
        data = [author_name, book_name, "{:.1f}%".format(percent)]
        writer.writerow(data)


def run_detection(author, books_dir_path):
    model = FakeBERTModel()
    #   loads the relevant model according to name and type
    model.load_model(TRAINED_MODELS_PATH + "Plagiarism" + "/" + author + ".h5")
    results_csv_path = author + '- Detection results_others.csv'
    for book_name in os.listdir(books_dir_path):
        if (book_name.lower().__contains__(".txt")):
            parts = book_name.lower().split('.');
            if parts[len(parts) - 1] == 'txt':
                book_path = books_dir_path + "/" + book_name;
                try:
                    book_content = read_book(book_path)

                    #   input preprocessing (separate input to chunks in size 128 tokens)
                    input_texts = get_preprocessed_text(book_content, max_text_len=model.config['max_seq_len'])

                    probabilities = model.predict(tf.constant(input_texts))

                    real_percent, fake_percent = get_distribution(probabilities)

                    write_results_to_file(file_path=results_csv_path, author_name=author,
                                          book_name=book_name,
                                          percent=real_percent)

                except:
                    print(book_path)
                    # break


# author_ = "Shakespeare"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_1\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)

# author_ = "William Shakespeare 2"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_2\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# author_ = "William Sheakspeare 3"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_3\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# author_ = "William Shakespeare 4"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_4\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)

# author_ = "Isaac Asimov"
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)

#
# author_ = "William Sheakspeare 3"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_3\\test\Others"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_1\\test\Others"
# run_detection(author=author_, books_dir_path=books_dir_path_)
#
# author_ = "William Shakespeare 4"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_4\\test\Others"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# author_ = "Shakespeare"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_1\\test\Others"
# run_detection(author=author_, books_dir_path=books_dir_path_)
#
# author_ = "William Shakespeare 2"
# books_dir_path_ = "DATABASE\plagiarism\Shakespeare_2\\test\Others"
# run_detection(author=author_, books_dir_path=books_dir_path_)


author_ = "Robert Sheckley 2"
books_dir_path_ = "DATABASE\plagiarism\RS\\test\\Others"
run_detection(author=author_, books_dir_path=books_dir_path_)
author_ = "Robert Sheckley"
books_dir_path_ = "DATABASE\plagiarism\RS\\test\\Others"
run_detection(author=author_, books_dir_path=books_dir_path_)

#
# author_ = "Isaac Asimov"
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# author_ = "Isaac Asimov"
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Benjamin Jonson"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Christopher Marlowe"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Francis Bacon"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Jean-Baptiste Poquelin"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Miguel de Cervantes Saavedra"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Niccolo Machiavelli"
# run_detection(author=author_, books_dir_path=books_dir_path_)
# books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_1\\test\\Others\\Robert Sheckley"
# run_detection(author=author_, books_dir_path=books_dir_path_)
author_ = "Isaac Asimov  2"
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others"
run_detection(author=author_, books_dir_path=books_dir_path_)
author_ = "Isaac Asimov  2"
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Benjamin Jonson"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Christopher Marlowe"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Francis Bacon"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Jean-Baptiste Poquelin"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Miguel de Cervantes Saavedra"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Niccolo Machiavelli"
run_detection(author=author_, books_dir_path=books_dir_path_)
books_dir_path_ = "DATABASE\plagiarism\Isaac Asimov_2\\test\\Others\\Robert Sheckley"
run_detection(author=author_, books_dir_path=books_dir_path_)
