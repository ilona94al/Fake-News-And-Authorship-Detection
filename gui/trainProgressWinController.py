from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class TrainProgressWinController(QMainWindow):
    def __init__(self ,author_name, folder_path, batch_size, epochs,parent=None):
        super(TrainProgressWinController, self).__init__(parent)
        from gui.train_progress_window import Ui_TrainProgressWindow
        self.ui = Ui_TrainProgressWindow()
        self.ui.setupUi(self)

        self.ui.resultsBtn.setEnabled(False)

        self.ui.resultsBtn.clicked.connect(self.results_pressed)
        self.ui.cancelBtn.clicked.connect(self.cancel_pressed)

        #model_manager(author name,path)
        # todo***: results=TRAIN MODEL (author name, path)

    def cancel_pressed(self):
        # todo: show pop up window
        self.close()
        from gui.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()

    def results_pressed(self):
        # todo***: results window and load the results from training
        # todo***:option to train again, or save the model

        self.close()
        from gui.detectionResultsWinController import DetectionResultsWinController
        self.window = DetectionResultsWinController()
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainProgressWinController()
    MainWindow.show()
    sys.exit(app.exec_())
