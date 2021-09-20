from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets

class DetectionResultsWinController(QMainWindow):
    def __init__(self, parent=None):
        super(DetectionResultsWinController, self).__init__(parent)
        # self.ui = Ui_DetectionResultsWindow()
        # self.ui.setupUi(self)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = DetectionResultsWinController()
    MainWindow.show()
    sys.exit(app.exec_())
