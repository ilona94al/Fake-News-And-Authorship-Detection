import threading
import time

from PyQt5 import QtWidgets
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from constants import RESOURCES_PATH


class TrainProgressWinController(QMainWindow):
    def __init__(self, task, parent=None):
        super(TrainProgressWinController, self).__init__(parent)
        self.task = task

        from gui_design.train_progress_window import Ui_TrainingProcessWindow
        self.ui = Ui_TrainingProcessWindow()
        self.ui.setupUi(self)

        self.ui.resultsBtn.setHidden(True)

        self.set_buttons_handlers()

        self.update_ui_with_data()

        self.task.start_train_model()

        self.t = threading.Thread(target=self.run_waiting)
        self.t.start()

    def set_buttons_handlers(self):
        self.ui.resultsBtn.clicked.connect(self.open_results_win)

    def update_ui_with_data(self):
        self.movie = QMovie("../" + RESOURCES_PATH + "please_wait3.gif")
        self.ui.loadinLabel.setMovie(self.movie)
        self.movie.start()

    def run_waiting(self):
        while self.task.running:
            time.sleep(10)
        # call to open_results_win()...
        self.ui.resultsBtn.click()

    def open_results_win(self):
        self.task.test_model()
        self.close()
        from gui_controllers.trainResultsWinController import TrainResultsWinController
        self.window = TrainResultsWinController(self.task)
        self.window.show()



