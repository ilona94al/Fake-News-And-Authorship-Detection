
import threading
import time

from PyQt5 import QtWidgets, QtGui,  Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from constants import RESOURCES_PATH
from gui_controllers.formCheckerWinController import FormCheckerWinController


class LoadingWinController(QMainWindow):
    def __init__(self, book, author, parent=None):
        super(LoadingWinController, self).__init__(parent)
        from gui_design.loading_window import Ui_LoadingWindow

        self.ui = Ui_LoadingWindow()
        self.ui.setupUi(self)


        self.set_disable_style(self.ui.resultsBtn)
        self.ui.resultsBtn.clicked.connect(self.results_pressed)
        self.ui.cancelBtn.clicked.connect(self.cancel_pressed)
        self.ui.cancelBtn.setHidden(True)

        self.movie = QMovie("../"+RESOURCES_PATH+"loading.gif")
        self.ui.loadinLabel.setMovie(self.movie)
        self.movie.start()

        self.loading = True

        self.t1 = threading.Thread(target=self.run_detection, args=(book, author))
        self.t1.start()

        self.t2 = threading.Thread(target=self.run_waiting)
        self.t2.start()

    def cancel_pressed(self):
        msg = QMessageBox()
        msg.setWindowTitle("Stop detection progress")
        msg.setText("Do you sure you want cancel detection?")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setInformativeText("The results will be lost")

        msg.buttonClicked.connect(self.PopUpAction)

        x = msg.exec_()

    def PopUpAction(self, i):
        if i.text() == 'OK':

            self.close()
            from gui_controllers.mainWinController import MainWindow
            self.window = MainWindow()
            self.window.show()


    def run_detection(self, book, author):
        from model.plagiarism_detection import PlagiarismDetection
        self.detection = PlagiarismDetection(input=book, model_name=author + ".h5", author_name=author)
        self.loading=False

    def run_waiting(self):
        while self.loading:
            time.sleep(10)

        pixmap=QtGui.QPixmap("../" + RESOURCES_PATH + "completed.png")
        pixmap_small = pixmap.scaled(64, 64)
        self.ui.loadinLabel.setPixmap(pixmap_small)
        self.ui.loadinLabel.adjustSize()
        self.set_enable_style(self.ui.resultsBtn)
        self.ui.cancelBtn.setHidden(False)


    def results_pressed(self):
        self.close()
        from gui_controllers.detectionResultsWinController import DetectionResultsWinController
        self.window = DetectionResultsWinController(self.detection)
        self.window.show()

    def set_enable_style(self, widget):
        widget.setEnabled(True)
        widget.setStyleSheet("background-color: rgb(0, 111, 0);\n"
                             "\n"
                             "font: 75 16pt \"Sitka Small\";\n"
                             "color: rgb(255, 255, 255);\n"
                             "border-width: 3px;\n"
                             "border-radius: 30px;\n"
                             "border-color: rgb(255, 255, 255);\n"
                             "border-style: solid;\n"
                             "\n"
                             "alternate-background-color: rgb(135, 135, 135);")
        widget.update()

    def set_disable_style(self, widget):
        widget.setEnabled(False)
        widget.setStyleSheet("background-color: rgb(0, 65, 0);\n"
                             "\n"
                             "font: 75 16pt \"Sitka Small\";\n"
                             "color: rgb(136, 136, 136);\n"
                             "border-width: 3px;\n"
                             "border-radius: 30px;\n"
                             "border-color: rgb(136, 136, 136);\n"
                             "border-style: solid;\n"
                             "\n"
                             "alternate-background-color: rgb(135, 135, 135);")
        widget.update()

