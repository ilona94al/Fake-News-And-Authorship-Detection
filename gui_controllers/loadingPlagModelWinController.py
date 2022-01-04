import os
import threading
import time

from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from constants import RESOURCES_PATH
from gui_controllers.loadingWinController import LoadingWinController


class LoadingPlagModelWinController(LoadingWinController):
    def __init__(self, book_name, content, author, parent=None):
        super(LoadingPlagModelWinController, self).__init__(parent)

        #   thread that loads the model, runs the model with input from user
        self.t1 = threading.Thread(target=self.run_detection, args=(book_name,content, author))
        self.t1.start()

        #   thread that waiting until first thread will finish
        #   and then raises event for the next step
        self.t2 = threading.Thread(target=self.run_waiting)
        self.t2.start()


    def run_detection(self, book_name, content, author):
        from model.plagiarism_detection import PlagiarismDetection
        os.chdir("../")
        self.detection = PlagiarismDetection(input=content, model_name=author + ".h5", author_name=author,book_name=book_name)
        os.chdir("gui_controllers")
        self.loading = False








