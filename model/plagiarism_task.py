import os

from constants import TRAINED_MODELS_PATH
from model.task import Task


class PlagiarismTask(Task):
    def __init__(self, author_name, dir_path, batch_size, epochs):
        super().__init__(batch_size, epochs)

        #   read books
        author_books = self.read_books_of_specific_author(books_dir_path=dir_path)
        different_books = self.read_books_of_various_authors(books_dir_path="../DATABASE/plagiarism/books_for_train1",
                                                             name_to_ignore=author_name)
        if (len(author_books) == 0):
            self.error = True
            self.error_msg = "Directory is empty, choose another."
            return
        else:
            self.error = False


        #   preprocessing
        author_texts = self.get_preprocessed_texts(author_books)
        diff_texts = self.get_preprocessed_texts(different_books)

        #   union
        texts = author_texts + diff_texts

        #   set original classification
        y_expected = self.define_expected_classification(len(author_texts), len(texts))

        #   split all data to train set, validation set and test set
        self.prepare_train_validation_test_sets(texts, y_expected)

        #   set probabilities for each text to belong for each label
        self.get_categorical_probabilities(self.y_test, self.y_train, self.y_valid)

        self.author = author_name

        self.model_path = TRAINED_MODELS_PATH + "Plagiarism/"

    #   gets author name and path to dir with books
    #   returns an array with books written by a specified author
    def read_books_of_specific_author(self, books_dir_path):
        books = []
        if len(os.listdir(books_dir_path)) != 0:
            for book_name in os.listdir(books_dir_path):
                if book_name.split('.')[1] == 'txt' or book_name.split('.')[1] == 'TXT':
                    book = self.read_book(books_dir_path, book_name)
                    books.append(book)
        return books

    #   gets author name and path to dir with books
    #   returns an array with books written by the different authors except for the specified author
    def read_books_of_various_authors(self, books_dir_path, name_to_ignore):
        books = []
        name_to_ignore_lower = name_to_ignore.lower()
        for author_name in os.listdir(books_dir_path):
            author_name_lower = author_name.lower()
            if not self.is_same_names(author_name_lower, name_to_ignore_lower):
                author_books = self.read_books_of_specific_author(books_dir_path + '/' + author_name)
                books.extend(book for book in author_books)
        return books

    @staticmethod
    def is_same_names(author_name, name_to_ignore):
        return (author_name == name_to_ignore) \
               or (author_name in name_to_ignore) \
               or (name_to_ignore in author_name)

    #   gets book name, author name, and path to dir with books
    #   returns book content as a string
    def read_book(self, books_dir_path, book_name):
        with open(books_dir_path + '/' + book_name, 'r', encoding='UTF-8') as book_file:
            book_string = book_file.read()
            return book_string
