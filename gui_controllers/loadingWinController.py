import threading
import time

from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from constants import RESOURCES_PATH

class LoadingWinController(QMainWindow):
    def __init__(self, parent=None):
        super(LoadingWinController, self).__init__(parent)
        from gui_design.loading_window import Ui_LoadingWindow

        self.ui = Ui_LoadingWindow()
        self.ui.setupUi(self)

        self.ui.resultsBtn.setHidden(True)

        self.set_buttons_handlers()

        self.update_ui_with_data()

        self.loading = True


    def update_ui_with_data(self):
        self.movie = QMovie("../" + RESOURCES_PATH + "please_wait3.gif")
        self.ui.loadinLabel.setMovie(self.movie)
        self.movie.start()

    def set_buttons_handlers(self):
        self.ui.resultsBtn.clicked.connect(self.next)




