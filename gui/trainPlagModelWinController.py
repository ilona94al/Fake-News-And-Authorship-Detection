from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainPlagModelWinController(QMainWindow):
    def __init__(self, parent=None):
        super(TrainPlagModelWinController, self).__init__(parent)
        from gui.train_plag_model_window import Ui_TrainPlagModelWindow
        self.ui = Ui_TrainPlagModelWindow()
        self.ui.setupUi(self)

        self.ui.radioButton1122.setChecked(True)
        self.ui.verticalGroupBox13.setHidden(False)
        self.ui.verticalGroupBox14.setHidden(True)


        self.ui.radioButton1122.clicked.connect(self.show_files_form)
        self.ui.radioButton1132.clicked.connect(self.show_folders_form)

        self.ui.errorMsg.setHidden(True)

        self.ui.uploadBtn1313.clicked.connect(lambda: self.upload_file_pressed(widget=self.ui.path1312))
        self.ui.uploadBtn1413.clicked.connect(lambda: self.upload_folder_pressed(widget=self.ui.path1412))

        self.ui.backBtn.clicked.connect(self.back_pressed)

    def show_scv_form(self):
        self.ui.verticalGroupBox13.setHidden(True)
        self.ui.verticalGroupBox14.setHidden(True)

    def show_files_form(self):

        self.ui.verticalGroupBox13.setHidden(False)
        self.ui.verticalGroupBox14.setHidden(True)

    def show_folders_form(self):
        self.ui.verticalGroupBox13.setHidden(True)
        self.ui.verticalGroupBox14.setHidden(False)

    def upload_file_pressed(self, widget):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        widget.setText(file_path)

    def upload_folder_pressed(self, widget):     # Folder uploader
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dir_path = filedialog.askdirectory()  # Returns opened path as str
        widget.setText(dir_path)              # Read path to field

    def back_pressed(self):
        self.close()
        from gui.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainPlagModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
