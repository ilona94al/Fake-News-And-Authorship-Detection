from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainModelWinController(QMainWindow):
    def __init__(self, parent=None):
        super(TrainModelWinController, self).__init__(parent)
        from gui.train_model_window import Ui_TrainModelWindow
        self.ui = Ui_TrainModelWindow()
        self.ui.setupUi(self)

        self.ui.verticalGroupBox12_1.setHidden(False)
        self.ui.verticalGroupBox12_2.setHidden(True)
        self.ui.verticalGroupBox12_3.setHidden(True)

        self.ui.radioButton1112.clicked.connect(self.show_scv_form)
        self.ui.radioButton1122.clicked.connect(self.show_files_form)
        self.ui.radioButton1132.clicked.connect(self.show_folders_form)

        self.ui.errorMsg.setHidden(True)

        self.ui.uploadBtn_1.clicked.connect(lambda: self.upload_pressed(widget=self.ui.path_1))
        self.ui.uploadBtn1_2.clicked.connect(lambda: self.upload_pressed(widget=self.ui.path1_2))
        self.ui.uploadBtn2_2.clicked.connect(lambda: self.upload_pressed(widget=self.ui.path2_2))
        self.ui.uploadBtn1_3.clicked.connect(lambda: self.upload_pressed(widget=self.ui.path1_3))
        self.ui.uploadBtn2_3.clicked.connect(lambda: self.upload_pressed(widget=self.ui.path2_3))

        self.ui.backBtn.clicked.connect(self.back_pressed)

    def show_scv_form(self):
        self.ui.verticalGroupBox12_1.setHidden(False)
        self.ui.verticalGroupBox12_2.setHidden(True)
        self.ui.verticalGroupBox12_3.setHidden(True)

    def show_files_form(self):
        self.ui.verticalGroupBox12_1.setHidden(True)
        self.ui.verticalGroupBox12_2.setHidden(False)
        self.ui.verticalGroupBox12_3.setHidden(True)

    def show_folders_form(self):
        self.ui.verticalGroupBox12_1.setHidden(True)
        self.ui.verticalGroupBox12_2.setHidden(True)
        self.ui.verticalGroupBox12_3.setHidden(False)

    def upload_pressed(self, widget):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()

        widget.setText(file_path)

    def back_pressed(self):
        self.close()
        from gui.plagiarismWinController import PlagiarismWinController
        self.window = PlagiarismWinController()
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
