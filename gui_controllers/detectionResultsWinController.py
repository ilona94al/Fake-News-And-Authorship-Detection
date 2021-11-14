import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class DetectionResultsWinController(QMainWindow):
    def __init__(self, detection,  parent=None):
        super(DetectionResultsWinController, self).__init__(parent)

        from gui_design.detection_results_window import Ui_DetectionResultsWindow
        self.ui = Ui_DetectionResultsWindow()
        self.ui.setupUi(self)

        detection.create_probabilities_plot()
        detection.create_distribution_plot()

        from PyQt5 import QtGui
        self.ui.plotLabel1.setPixmap(QtGui.QPixmap("../"+detection.plot_path1))
        self.ui.plotLabel2.setPixmap(QtGui.QPixmap("../"+detection.plot_path2))

        self.ui.backBtn.clicked.connect(self.back_pressed)

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()



