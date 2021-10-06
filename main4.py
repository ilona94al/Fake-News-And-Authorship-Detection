import os
import re

from nltk import word_tokenize
from nltk.corpus import stopwords


def read_books_of_specific_author(author_name):
    res = []

    for book_name in os.listdir(root_books_dir_path + '/' + author_name):
        if book_name.split('.')[1] == 'txt':
            res.append(read_book(author_name, book_name))

    return res


def read_books_of_various_authors(name_to_ignore):
    res = []

    for author_name in os.listdir(root_books_dir_path):
        if author_name != name_to_ignore:
            author_books = read_books_of_specific_author(author_name)
            res.extend(book for book in author_books)

    return res


def read_book(writer_name, book_name):
    with open(root_books_dir_path + '/' + writer_name + '/' + book_name, 'r', encoding='UTF-8') as book_file:
        book_string = book_file.read()
        filtered_text_str = text_preprocessing(book_string)
        return filtered_text_str


def text_preprocessing(string):
    regex_legal_letters = re.compile('[^ \\t\\n\\v\\f\\ra-zA-Z]')
    string = regex_legal_letters.sub('', string)
    string = string.lower()
    book_words_arr = string.split()
    arr = remove_stop_words(book_words_arr)
    filtered_text_str = " ".join(word for word in arr)
    return filtered_text_str


def remove_stop_words(word_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return filtered_text


root_books_dir_path = 'DB/books'

author_books = read_books_of_specific_author(author_name='shakespeare')
different_books = read_books_of_various_authors(name_to_ignore='shakespeare')


books = author_books + different_books

# define original classification
y_expected = define_expected_classification(len(author_books), len(different_books))

