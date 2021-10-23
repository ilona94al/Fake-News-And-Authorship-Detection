from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainPlagModelWinController(QMainWindow):
    def __init__(self, parent=None):
        super(TrainPlagModelWinController, self).__init__(parent)
        from gui.train_plag_model_window import Ui_TrainPlagModelWindow
        self.ui = Ui_TrainPlagModelWindow()
        self.ui.setupUi(self)

        self.ui.errorMsg.setHidden(True)

        self.ui.uploadBtn1112.clicked.connect(lambda: self.upload_file_pressed(widget=self.ui.inputPath1111))

        self.ui.backBtn.clicked.connect(self.back_pressed)


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
