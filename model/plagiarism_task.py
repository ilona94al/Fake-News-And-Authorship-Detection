from model.task import Task


class PlagiarismTask(Task):
    def __init__(self, author_name, dir_path):
        super().__init__()

        #   read books
        author_books = self.read_books_of_specific_author(books_dir_path=dir_path, author_name=author_name)
        different_books = self.read_books_of_various_authors(books_dir_path=dir_path, name_to_ignore=author_name)

        #   preprocessing
        author_texts = self.get_preprocessed_texts(author_books)
        diff_texts = self.get_preprocessed_texts(different_books)

        #   union
        texts = author_texts + diff_texts

        #   set original classification
        y_expected = self.define_expected_classification(len(author_texts), len(texts))

        #   split all data to train set, validation set and test set
        x_train, x_valid, x_test, y_train, y_valid, y_test = \
            self.prepare_train_validation_test_sets(texts, y_expected)

        #   set probabilities for each text to belong for each label
        #   ----    Real    |   Fake
        #   text1   0.0     |   1.0
        #   text1   1.0     |   0.0
        #   ....    ...         ...
        y_train_prob, y_valid_prob, y_test_prob = self.get_categorical_probabilities(y_test, y_train, y_valid)

        num_classes = y_train_prob.shape[1]

    def read_books_of_specific_author(self, books_dir_path, author_name):
        books = []
        for book_name in os.listdir(books_dir_path + '/' + author_name):
            if book_name.split('.')[1] == 'txt':
                book = self.read_book(books_dir_path, author_name, book_name)
                books.append(book)
        return books

    def read_books_of_various_authors(self, books_dir_path, name_to_ignore):
        books = []
        for author_name in os.listdir(books_dir_path):
            if author_name != name_to_ignore:
                author_books = self.read_books_of_specific_author(books_dir_path, author_name)
                books.extend(book for book in author_books)
        return books

    def read_book(self, books_dir_path, writer_name, book_name):
        with open(books_dir_path + '/' + writer_name + '/' + book_name, 'r', encoding='UTF-8') as book_file:
            book_string = book_file.read()
            return book_string