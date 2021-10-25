from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainPlagModelWinController(QMainWindow):
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

    def back_pressed(self):
        self.close()
        from gui.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()

    def train_pressed(self):
        self.clear_feedback()
        # todo: open training process window (need to create ui,py,controller)
        # todo***: results=TRAIN MODEL (author name, path)
        # todo***: results window and load the results from training
        # todo***:option to train again, or save the model

        folder_path = str(self.ui.inputPath1131.text())
        author_name = str(self.ui.inputAuthorName111.text())
        epochs = str(self.ui.inputEpoch1152.text())
        batch_size = str(self.ui.inputBatchSz1154.text())

        if folder_path == "":
            self.folder_not_uploaded("Please specify path to folder")
        if author_name == "":
            self.no_author_name("Please fill author name")
        if batch_size == "" or not batch_size.isnumeric() or int(batch_size) <= 0:
            self.no_batch_size("Please fill valid batch size")
        if epochs == "" or not epochs.isnumeric() or int(epochs) <= 0:
            self.no_epochs("Please fill number of epochs")
        if self.allOk == True:
            self.close()
            from gui.trainProgressWinController import TrainProgressWinController
            self.window = TrainProgressWinController(author_name, folder_path, batch_size, epochs)
            self.window.show()

    def clear_feedback(self):
        self.allOk = True
        self.ui.errorMsg.setHidden(True)
        self.ui.errorMsg.setText("")
        self.set_normal_style(self.ui.inputPath1131)
        self.set_normal_style(self.ui.inputAuthorName111)
        self.set_normal_style(self.ui.inputEpoch1152)
        self.set_normal_style(self.ui.inputBatchSz1154)

    def folder_not_uploaded(self, msg):
        widget = self.ui.inputPath1131
        self.setFeedback(msg, widget)

    def no_author_name(self, msg):
        widget = self.ui.inputAuthorName111
        self.setFeedback(msg, widget)

    def no_batch_size(self, msg):
        widget = self.ui.inputBatchSz1154
        self.setFeedback(msg, widget)

    def no_epochs(self, msg):
        widget = self.ui.inputEpoch1152
        self.setFeedback(msg, widget)

    def setFeedback(self, msg, widget):
        self.allOk = False
        curr_msg = self.ui.errorMsg.text()
        if curr_msg != "":
            curr_msg += "\n"
        self.ui.errorMsg.setText(curr_msg + msg)
        self.ui.errorMsg.adjustSize()
        self.ui.errorMsg.setHidden(False)
        self.set_error_style(widget)

    def set_normal_style(self, widget):
        widget.setStyleSheet("border-width: 3px;\n"
                             "                                                    border-radius: 5px;\n"
                             "                                                    border-color: rgb(0, 0, 0);\n"
                             "                                                    border-style: solid;\n"
                             "                                                    background-color: rgb(188, 188, 188);\n"
                             "                                                    color: rgb(0, 0, 0);\n"
                             "                                                \n"
                             "")
        widget.update()

    def set_error_style(self, widget):
        widget.setStyleSheet("border-width: 3px;\n"
                             "                                                    border-radius: 5px;\n"
                             "                                                    border-color: rgb(170, 0, 0);\n"
                             "                                                    border-style: solid;\n"
                             "                                                    background-color: rgb(188, 188, 188);\n"
                             "                                                    color: rgb(0, 0, 0);\n"
                             "                                                \n"
                             "")
        widget.update()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainPlagModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
