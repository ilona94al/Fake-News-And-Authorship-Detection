
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow

from gui.mainWinController import MainWinController
from gui.plagiarism_window import Ui_PlagiarismWindow
from gui.trainModelWinController import TrainModelWinController


class PlagiarismWinController(QMainWindow):
    def __init__(self, parent=None):
        super(PlagiarismWinController, self).__init__(parent)
        self.ui = Ui_PlagiarismWindow()
        self.ui.setupUi(self)
        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.uploadBtn.clicked.connect(self.upload_pressed)
        self.ui.trainBtn.clicked.connect(self.train_pressed)
        self.ui.startBtn.clicked.connect(self.start_pressed)


        self.ui.errorMsg.setHidden(True)


        # todo: upload writers name into a combo box.
        self.ui.authorComboBox.addItem("")
        self.ui.authorComboBox.addItem("Vlad")

    def back_pressed(self):
        self.close()
        self.window = MainWinController()
        self.window.show()

    def upload_pressed(self):
        import tkinter as tk
        from tkinter import filedialog

        root=tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()

        self.ui.path.setText(file_path)

    def train_pressed(self):
        self.close()
        self.window = TrainModelWinController()
        self.window.show()

    def start_pressed(self):
        self.clear_feedback()

        book_path = self.ui.path.text()
        author_name = self.ui.authorComboBox.currentText()
        if str(book_path) == "":
            self.book_not_uploaded()
        if str(author_name) == "":
            self.author_not_chosen()
        # self.close()
        # self.window = DetectionResultsWinController()
        # self.window.show()

    def clear_feedback(self):
        self.ui.errorMsg.setHidden(True)
        self.ui.errorMsg.setText("")
        self.ui.horizontalGroupBox.setStyleSheet("")
        self.ui.verticalGroupBox.setStyleSheet("")

    def book_not_uploaded(self):
        self.ui.errorMsg.setText("Empty path, please upload a book.")
        self.ui.errorMsg.setHidden(False)
        self.ui.errorMsg.adjustSize()
        self.ui.horizontalGroupBox.setStyleSheet("border-color: rgb(170, 0, 0);")
        self.ui.horizontalGroupBox.update()

    def author_not_chosen(self):
        self.ui.errorMsg.setText(self.ui.errorMsg.text() + "\n" + "Author not chosen")
        self.ui.errorMsg.setHidden(False)
        self.ui.errorMsg.adjustSize()
        self.ui.verticalGroupBox.setStyleSheet("border-color: rgb(170, 0, 0);")
        self.ui.verticalGroupBox.update()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PlagiarismWinController()
    MainWindow.show()
    sys.exit(app.exec_())
