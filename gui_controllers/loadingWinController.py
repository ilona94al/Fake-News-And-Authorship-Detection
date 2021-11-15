import threading
import time

from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from constants import RESOURCES_PATH

class LoadingWinController(QMainWindow):
    def __init__(self, book_name, content, author, parent=None):
        super(LoadingWinController, self).__init__(parent)
        from gui_design.loading_window import Ui_LoadingWindow

        self.ui = Ui_LoadingWindow()
        self.ui.setupUi(self)

        self.ui.resultsBtn.setHidden(True)

        self.set_buttons_handlers()

        self.update_ui_with_data()

        self.loading = True

        #   thread that loads the model, runs the model with input from user
        self.t1 = threading.Thread(target=self.run_detection, args=(book_name,content, author))
        self.t1.start()

        #   thread that waiting until first thread will finish
        #   and then raises event for the next step
        self.t2 = threading.Thread(target=self.run_waiting)
        self.t2.start()

    def update_ui_with_data(self):
        self.movie = QMovie("../" + RESOURCES_PATH + "please_wait3.gif")
        self.ui.loadinLabel.setMovie(self.movie)
        self.movie.start()

    def set_buttons_handlers(self):
        self.ui.resultsBtn.clicked.connect(self.next)


    def run_detection(self, book_name, content, author):
        from model.plagiarism_detection import PlagiarismDetection
        self.detection = PlagiarismDetection(input=content, model_name=author + ".h5", author_name=author,book_name=book_name)
        self.loading = False

    def run_waiting(self):
        while self.loading:
            time.sleep(10)
        self.ui.resultsBtn.click()

    def next(self):
        self.close()
        from gui_controllers.detectionResultsWinController import DetectionResultsWinController
        self.window = DetectionResultsWinController(self.detection)
        self.window.show()


