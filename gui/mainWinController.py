from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets




class MainWinController(QMainWindow):
    def __init__(self, parent=None):
        super(MainWinController, self).__init__(parent)
        from gui.main_window import Ui_MainWindow
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.plagiarismBtn.clicked.connect(self.openPlagiarisWin)

    def openPlagiarisWin(self):
        self.close()
        from gui.plagiarismWinController import PlagiarismWinController
        self.window = PlagiarismWinController()
        self.window.show()
        # self.window = QtWidgets.QMainWindow()
        # self.ui = Ui_PlagiarismWindow()
        # self.ui.setupUi(self.window)
        # self.window.show()
        # MainWindow.hide()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWinController()
    MainWindow.show()
    sys.exit(app.exec_())
