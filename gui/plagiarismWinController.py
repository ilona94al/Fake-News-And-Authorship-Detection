from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class PlagiarismWinController(QMainWindow):
    def __init__(self, parent=None):
        super(PlagiarismWinController, self).__init__(parent)
        from gui.plagiarism_window import Ui_PlagiarismWindow

        self.ui = Ui_PlagiarismWindow()
        self.ui.setupUi(self)

        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.uploadBtn.clicked.connect(self.upload_pressed)
        self.ui.startBtn.clicked.connect(self.start_pressed)

        self.ui.errorMsg.setHidden(True)

        # todo: upload writers name into a combo box.

        self.ui.authorComboBox.addItem("Vlad")

    def back_pressed(self):
        self.close()
        from gui.mainWinController import MainWinController
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

        book_path = self.ui.inputPath.text()

        author_name = self.ui.authorComboBox.currentText()
        if str(book_path) == "":
            self.book_not_uploaded( "Empty path!\n please upload a book")
        else:
            # todo: check if book in .txt format. if no - error message
            #self.book_not_uploaded( "Book not in .txt format")
            self.close()
            from gui.detectionResultsWinController import DetectionResultsWinController
            # todo: results= DETECT(author_name, tweet)
            #  find the relevant trained model(according to the author name)
            #  insert book as input to the model and get detection results (graphs and etc.)
            self.window = DetectionResultsWinController(author_name, book_path)
            self.window.show()

    def clear_feedback(self):
        self.ui.errorMsg.setHidden(True)
        self.ui.errorMsg.setText("")
        self.ui.horizontalGroupBox.setStyleSheet("")

    def book_not_uploaded(self,msg):
        widget = self.ui.horizontalGroupBox
        self.setFeedback(msg, widget)

    def setFeedback(self, msg, widget):
        self.ui.errorMsg.setText(msg)
        self.ui.errorMsg.adjustSize()
        self.ui.errorMsg.setHidden(False)
        widget.setStyleSheet("border-color: rgb(170, 0, 0);")
        widget.update()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PlagiarismWinController()
    MainWindow.show()
    sys.exit(app.exec_())
