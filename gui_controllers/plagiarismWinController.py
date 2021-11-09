import os

from PyQt5 import QtWidgets

from gui_controllers.formCheckerWinController import FormCheckerWinController


class PlagiarismWinController(FormCheckerWinController):
    def __init__(self, parent=None):
        super(PlagiarismWinController, self).__init__(parent)
        from gui_design.plagiarism_window import Ui_PlagiarismWindow

        self.ui = Ui_PlagiarismWindow()
        self.ui.setupUi(self)

        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.uploadBtn.clicked.connect(self.upload_pressed)
        self.ui.startBtn.clicked.connect(self.start_pressed)

        self.ui.authorComboBox.clear()
        os.chdir("../TRAINED_MODELS/")
        arr = os.listdir('Plagiarism')
        models=[]
        for item in arr:
            if item.split(".")[1]=="h5":
                models.append(item)
        self.ui.authorComboBox.addItems(models)
        os.chdir("../gui_controllers")


        self.clear_feedback()

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()

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


        book_path = path_widget.text()

        author_name = self.ui.authorComboBox.currentText()
        if str(book_path) == "":
            self.invalid_input("Empty path!\n please upload a book", path_widget)
        else:
            self.set_normal_style(path_widget)
            # todo: check if book in .txt format. if no - error message
            #self.invalid_input( "Book not in .txt format",self.gui_design.inputPath)
            self.close()
            from gui_controllers.detectionResultsWinController import DetectionResultsWinController
            # todo: results= DETECT(author_name, tweet)
            #  find the relevant trained model(according to the author name)
            #  insert book as input to the model and get detection results (graphs and etc.)
            self.window = DetectionResultsWinController(author_name, book_path)
            self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PlagiarismWinController()
    MainWindow.show()
    sys.exit(app.exec_())
