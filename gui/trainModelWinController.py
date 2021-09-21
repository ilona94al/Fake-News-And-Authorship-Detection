from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainModelWinController(QMainWindow):
    def __init__(self, parent=None):
        super(TrainModelWinController, self).__init__(parent)
        # self.ui = Ui_TrainModelWindow()
        # self.ui.setupUi(self)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
