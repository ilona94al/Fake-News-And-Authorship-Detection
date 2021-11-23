import os

from PyQt5 import QtWidgets

from constants import TRAINED_MODELS_PATH
from gui_controllers.formCheckerWinController import FormCheckerWinController


class PlagiarismWinController(FormCheckerWinController):
    def __init__(self, parent=None):
        super(PlagiarismWinController, self).__init__(parent)
        from gui_design.plagiarism_window import Ui_PlagiarismWindow
        self.ui = Ui_PlagiarismWindow()
        self.ui.setupUi(self)

        self.set_buttons_handlers()

        self.update_ui_with_data()

        self.clear_feedback()

    def update_ui_with_data(self):
        self.ui.authorComboBox.clear()
        arr = os.listdir('../' + TRAINED_MODELS_PATH + 'Plagiarism')
        models = []
        for item in arr:
            if item.split(".")[1] == "h5":
                models.append(item.removesuffix(".h5"))
        self.ui.authorComboBox.addItems(models)

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.uploadBtn.clicked.connect(self.upload_pressed)
        self.ui.startBtn.clicked.connect(self.start_pressed)

    def upload_pressed(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        self.ui.inputPath.setText(file_path)

    def start_pressed(self):
        self.clear_feedback()

        path_widget = self.ui.inputPath

        self.set_normal_style(path_widget)

        book_path = path_widget.text()

        author_name = self.ui.authorComboBox.currentText()
        if str(book_path) == "":
            self.invalid_input("Empty path!\n please upload a book", path_widget)
        elif book_path.split(".")[1] != "txt":
            self.invalid_input("Book not in .txt format", path_widget)
        else:
            self.set_normal_style(path_widget)

            book_content, book_name = self.read_book(book_path)

            self.close()
            from gui_controllers.loadingPlagModelWinController import LoadingPlagModelWinController
            self.window = LoadingPlagModelWinController(book_name=book_name, content=book_content, author=author_name)
            self.window.show()

    @staticmethod
    def read_book(book_dir_path):
        path_parts = book_dir_path.split("/")
        last_part_i = len(path_parts)
        book_name = path_parts[last_part_i - 1].removesuffix(".txt")
        with open(book_dir_path, 'r', encoding='UTF-8') as book_file:
            book_string = book_file.read()
            return book_string, book_name


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PlagiarismWinController()
    MainWindow.show()
    sys.exit(app.exec_())
