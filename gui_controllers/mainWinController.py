from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class MainWinController(QMainWindow):
    def __init__(self, parent=None):
        super(MainWinController, self).__init__(parent)
        from gui_design.main_window import Ui_MainWindow
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.set_buttons_handlers()

    def set_buttons_handlers(self):
        self.ui.plagiarismBtn.clicked.connect(self.open_plagiarism_win)
        self.ui.fakeNewsBtn.clicked.connect(self.open_fake_news_win)
        self.ui.trainBtn.clicked.connect(self.train_pressed)

    def open_plagiarism_win(self):
        self.close()
        from gui_controllers.plagiarismWinController import PlagiarismWinController
        self.window = PlagiarismWinController()
        self.window.show()

    def open_fake_news_win(self):
        self.close()
        from gui_controllers.fakeNewsWinController import FakeNewsWinController
        self.window = FakeNewsWinController()
        self.window.show()

    def train_pressed(self):
        self.close()
        from gui_controllers.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWinController()
    MainWindow.show()
    sys.exit(app.exec_())
