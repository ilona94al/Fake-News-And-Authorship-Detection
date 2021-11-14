import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow

from constants import RESOURCES_PATH


class DetectionResultsWinController(QMainWindow):
    def __init__(self, detection, parent=None):
        super(DetectionResultsWinController, self).__init__(parent)

        from gui_design.detection_results_window import Ui_DetectionResultsWindow
        self.ui = Ui_DetectionResultsWindow()
        self.ui.setupUi(self)

        detection.create_probabilities_plot()
        detection.create_distribution_plot()

        from PyQt5 import QtGui
        pixmap = QtGui.QPixmap("../" + detection.plot_path1)
        pixmap_small = pixmap.scaled(490, 490)
        self.ui.plotLabel1.setPixmap(pixmap_small)

        pixmap = QtGui.QPixmap("../" + detection.plot_path2)
        pixmap_small = pixmap.scaled(490, 490)
        self.ui.plotLabel2.setPixmap(pixmap_small)

        text = "A model detects that the book: " + detection.book_name + " was written by " + detection.author_name \
               + " with " + detection.real_percent + " percent certainty." + "\n"
        self.ui.resultLabel.setText(text)
        self.ui.backBtn.clicked.connect(self.back_pressed)

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()
