from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class MainWinController(QMainWindow):
    def __init__(self, parent=None):
        super(MainWinController, self).__init__(parent)
        from gui.ui.main_window import Ui_MainWindow
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.plagiarismBtn.clicked.connect(self.openPlagiarismWin)
        self.ui.fakeNewsBtn.clicked.connect(self.openFakeNewsWin)

        self.ui.trainBtn.clicked.connect(self.train_pressed)

    def openPlagiarismWin(self):
        self.close()
        from gui.controllers.plagiarismWinController import PlagiarismWinController
        self.window = PlagiarismWinController()
        self.window.show()
    def openFakeNewsWin(self):
        self.close()
        from gui.controllers.fakeNewsWinController import FakeNewsWinController
        self.window = FakeNewsWinController()
        self.window.show()

    def train_pressed(self):
        self.close()
        from gui.controllers.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWinController()
    MainWindow.show()
    sys.exit(app.exec_())
