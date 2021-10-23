from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainNewsModelWinController(QMainWindow):
    def __init__(self, parent=None):
        super(TrainNewsModelWinController, self).__init__(parent)
        from gui.train_news_model_window import Ui_TrainNewsModelWindow
        self.ui = Ui_TrainNewsModelWindow()
        self.ui.setupUi(self)

        self.ui.radioButton2111.setChecked(True)
        self.ui.verticalGroupBox11.setHidden(False)
        self.ui.verticalGroupBox12.setHidden(True)

        self.ui.radioButton2111.clicked.connect(self.show_one_file_form)
        self.ui.radioButton2121.clicked.connect(self.show_two_files_form)

        self.ui.errorMsg.setHidden(True)

        self.ui.uploadBtn1212.clicked.connect(lambda: self.upload_file_pressed(widget=self.ui.inputPath1211))
        self.ui.uploadBtn1112.clicked.connect(lambda: self.upload_file_pressed(widget=self.ui.inputPath1111))
        self.ui.uploadBtn1122.clicked.connect(lambda: self.upload_file_pressed(widget=self.ui.inputPath1121))

        self.ui.backBtn.clicked.connect(self.back_pressed)

    def show_one_file_form(self):
        self.ui.verticalGroupBox11.setHidden(False)
        self.ui.verticalGroupBox12.setHidden(True)

    def show_two_files_form(self):
        self.ui.verticalGroupBox11.setHidden(True)
        self.ui.verticalGroupBox12.setHidden(False)

    def upload_file_pressed(self, widget):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        widget.setText(file_path)

    def upload_folder_pressed(self, widget):  # Folder uploader
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dir_path = filedialog.askdirectory()  # Returns opened path as str
        widget.setText(dir_path)  # Read path to field

    def back_pressed(self):
        self.close()
        from gui.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()

    def train_pressed(self):
        self.close()
        # todo: add parametrs for train, epochs, batch size...
        # todo: check author name and path is not empty - if needed show erorr message
        # todo: open training process window (need to create ui,py,controller)
        # todo***: results=TRAIN MODEL (author name, path)
        # todo***: open results window and load the results from training
        # todo***:option to train again, or save the model


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainNewsModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
