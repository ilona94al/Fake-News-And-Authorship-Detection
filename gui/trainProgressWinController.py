from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox


class TrainProgressWinController(QMainWindow):
    def __init__(self, task, parent=None):
        super(TrainProgressWinController, self).__init__(parent)
        self.task = task
        from gui.train_progress_window import Ui_TrainProgressWindow
        self.ui = Ui_TrainProgressWindow()
        self.ui.setupUi(self)

        self.ui.resultsBtn.setEnabled(False)

        self.ui.resultsBtn.clicked.connect(self.results_pressed)
        self.ui.cancelBtn.clicked.connect(self.cancel_pressed)
        self.ui.progressBar
        #  todo: progress percent will calculated by:
        #              ?.passed epochs / task.number of epochs
        #   - can we do it ?!!
        # todo: get notification that fit thread finish -> set enable results button

    def cancel_pressed(self):
        msg = QMessageBox()
        msg.setWindowTitle("Train new model")
        msg.setText("Do you sure to stop training progress?")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setInformativeText("The process will be reset!")

        msg.buttonClicked.connect(self.PopUpAction)
        x = msg.exec_()

    def PopUpAction(self, i):
        if i.text() == 'OK':
            self.close()
            from gui.chooseTrainWinController import ChooseTrainWinController
            self.window = ChooseTrainWinController()
            self.window.show()

    def results_pressed(self):
        # todo: create results window and load the results from training
        #  (task.accuracy, test params)
        # todo**: option to train again, or save the model - task.save_model(name)

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
