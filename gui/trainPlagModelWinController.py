from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow

from gui.trainModelWinController import TrainModelWinController


class TrainPlagModelWinController(TrainModelWinController):
    def __init__(self, parent=None):
        super(TrainPlagModelWinController, self).__init__(parent)
        from gui.train_plag_model_window import Ui_TrainPlagModelWindow
        self.ui = Ui_TrainPlagModelWindow()
        self.ui.setupUi(self)

        self.ui.errorMsg.setHidden(True)

        self.ui.uploadBtn1132.clicked.connect(lambda: self.upload_folder_pressed(widget=self.ui.inputPath1131))
        self.ui.trainBtn.clicked.connect(self.train_pressed)
        self.ui.backBtn.clicked.connect(self.back_pressed)

    def upload_folder_pressed(self, widget):  # Folder uploader
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dir_path = filedialog.askdirectory()  # Returns opened path as str
        widget.setText(dir_path)  # Read path to field

    def train_pressed(self):
        self.clear_feedback()
        # todo: open training process window (need to create ui,py,controller)
        # todo***: results=TRAIN MODEL (author name, path)
        # todo***: results window and load the results from training
        # todo***:option to train again, or save the model

        folder_path_widget = self.ui.inputPath1131
        author_name_widget = self.ui.inputAuthorName111
        epochs_widget = self.ui.inputEpoch1152
        batch_size_widget = self.ui.inputBatchSz1154

        folder_path = str(folder_path_widget.text())
        author_name = str(author_name_widget.text())
        epochs = str(epochs_widget.text())
        batch_size = str(batch_size_widget.text())

        if folder_path == "":
            self.invalid_input("path to folder is miss", folder_path_widget)
        else:
            self.set_normal_style(folder_path_widget)
        if author_name == "":
            self.invalid_input("Please fill author name is miss", author_name_widget)
        else:
            self.set_normal_style(author_name_widget)
        if batch_size == "" or not batch_size.isnumeric() or int(batch_size) <= 0:
            self.invalid_input("Please fill valid batch size", batch_size_widget)
        else:
            self.set_normal_style(batch_size_widget)
        if epochs == "" or not epochs.isnumeric() or int(epochs) <= 0:
            self.invalid_input("Please fill valid number of epochs", epochs_widget)
        else:
            self.set_normal_style(epochs_widget)
        if self.allOk == True:
            self.close()
            from model.plagiarism_task import PlagiarismTask
            self.task = PlagiarismTask(author_name, folder_path, int(batch_size), int(epochs))
            self.next()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainPlagModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
