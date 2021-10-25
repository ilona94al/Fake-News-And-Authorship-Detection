import os
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split


class DataManager():
    def __init__(self, author_name, dir_path):

        self.author_books = self.read_books_of_specific_author(books_path=dir_path, author_name=author_name)
        self.different_books = self.read_books_of_various_authors(books_path=dir_path, name_to_ignore=author_name)

        author_texts = []
        for book in author_books:
            preprocessed_text = text_preprocessing(book)
            book_blocks = seperate_to_blocks(preprocessed_text)
            author_texts.extend(block for block in book_blocks)
        diff_texts = []
        for book in different_books:
            preprocessed_text = text_preprocessing(book)
            book_blocks = seperate_to_blocks(preprocessed_text)
            diff_texts.extend(block for block in book_blocks)

        texts = author_texts + diff_texts

        # define original classification
        y_expected = define_expected_classification(len(author_texts), len(texts))

        x_train, x, y_train, y = train_test_split(texts, y_expected, train_size=0.7)
        x_test, x_valid, y_test, y_valid = train_test_split(x, y, train_size=0.5)

        y_train_prob = np_utils.to_categorical(y_train)
        y_test_prob = np_utils.to_categorical(y_test)
        y_valid_prob = np_utils.to_categorical(y_valid)
        num_classes = y_train_prob.shape[1]

    def read_books_of_specific_author(self,books_path, author_name):
        books = []
        for book_name in os.listdir(books_path + '/' + author_name):
            if book_name.split('.')[1] == 'txt':
                book = self.read_book(books_path, author_name, book_name)
                books.append(book)
        return books

    def read_books_of_various_authors(self,books_path, name_to_ignore):
        books = []
        for author_name in os.listdir(books_path):
            if author_name != name_to_ignore:
                author_books = self.read_books_of_specific_author(books_path, author_name)
                books.extend(book for book in author_books)
        return books

    def read_book(self,books_dir_path, writer_name, book_name):
        with open(books_dir_path + '/' + writer_name + '/' + book_name, 'r', encoding='UTF-8') as book_file:
            book_string = book_file.read()
            return book_string
