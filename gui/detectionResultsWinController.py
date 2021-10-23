from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class DetectionResultsWinController(QMainWindow):
    def __init__(self,  parent=None):
        super(DetectionResultsWinController, self).__init__(parent)

        # todo: show results on ui
        from gui.detection_results_window import Ui_DetectionResultsWindow
        self.ui = Ui_DetectionResultsWindow()
        self.ui.setupUi(self)
        self.ui.backBtn.clicked.connect(self.back_pressed)

    def back_pressed(self):
        self.close()
        from gui.plagiarismWinController import PlagiarismWinController
        self.window = PlagiarismWinController()
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = DetectionResultsWinController()
    MainWindow.show()
    sys.exit(app.exec_())
