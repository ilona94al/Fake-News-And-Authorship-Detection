import threading
import time

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox


class TrainProgressWinController(QMainWindow):
    def __init__(self, task, parent=None):
        super(TrainProgressWinController, self).__init__(parent)
        self.task = task
        from gui_design.train_progress_window import Ui_TrainProgressWindow
        self.ui = Ui_TrainProgressWindow()
        self.ui.setupUi(self)

        self.set_disable_style(self.ui.resultsBtn)
        self.ui.resultsBtn.clicked.connect(self.results_pressed)
        self.ui.cancelBtn.clicked.connect(self.cancel_pressed)
        self.ui.cancelBtn.setHidden(True)

        self.task.start_train_model(self.ui.progressBar)

        self.t = threading.Thread(target=self.run_check_running)
        self.t.start()

    def run_check_running(self):
        while self.task.running:
            time.sleep(10)
        self.set_enable_style(self.ui.resultsBtn)
        self.ui.cancelBtn.setHidden(False)

    def cancel_pressed(self):
        msg = QMessageBox()
        msg.setWindowTitle("Train new model")
        msg.setText("Do you sure you want cancel progress?")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setInformativeText("The model will be lose")

        msg.buttonClicked.connect(self.PopUpAction)

        x = msg.exec_()

    def PopUpAction(self, i):
        if i.text() == 'OK':
            # todo: stop all threads
            self.task.stop_thread()

            self.close()
            from gui_controllers.chooseTrainWinController import ChooseTrainWinController
            self.window = ChooseTrainWinController()
            self.window.show()

    def results_pressed(self):
        # todo: create results window and load the results from training
        #  (task.accuracy, test params)
        # todo**: option to train again, or save the model - task.save_model(name)
        self.task.test_model()
        self.close()
        from gui_controllers.trainResultsWinController import TrainResultsWinController
        self.window = TrainResultsWinController(self.task)
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


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainProgressWinController()
    MainWindow.show()
    sys.exit(app.exec_())
