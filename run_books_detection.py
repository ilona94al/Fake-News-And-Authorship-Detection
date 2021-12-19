import os

from model.plagiarism_detection import PlagiarismDetection


def read_book(book_dir_path):
    path_parts = book_dir_path.split("/")
    last_part_i = len(path_parts)
    book_name = path_parts[last_part_i - 1].removesuffix(".txt")
    with open(book_dir_path, 'r', encoding='UTF-8') as book_file:
        book_string = book_file.read()
        return book_string, book_name


author = "Isaac Asimov"
books_dir_path = "DATABASE\plagiarism\Isaac Asimov_1\\test"
for book_name in os.listdir(books_dir_path):
    if (book_name.lower().__contains__(".txt")):
        parts = book_name.lower().split('.');
        if parts[len(parts) - 1] == 'txt':
            book_path = books_dir_path + "/" + book_name;
            try:
                book_content, book_name = read_book(book_path)
                PlagiarismDetection(input=book_content, model_name=author + ".h5", author_name=author,
                                    book_name=book_name)

            except:
                print(book_path)
                # break
