
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from constants import PLOTS_PATH
from PyQt5 import QtGui


class TrainResultsWinController(QMainWindow):
    def __init__(self, task, parent=None):
        super(TrainResultsWinController, self).__init__(parent)
        self.task = task

        from gui_design.train_results_window import Ui_TrainResultsWindow
        self.ui = Ui_TrainResultsWindow()
        self.ui.setupUi(self)

        self.set_buttons_handlers()

        self.update_ui_with_data()

    def update_ui_with_data(self):
        self.ui.accuracyGraph.setPixmap(QtGui.QPixmap("../" + PLOTS_PATH + "ModelAcc.png"))
        self.ui.accuracyGraph.setPixmap(QtGui.QPixmap("../" + PLOTS_PATH + "ModelAcc.png"))
        results = "Number of true predicts: " + str(self.task.model.count_well_predicted) + "\n" \
                  + "Number of false predicts: " + str(self.task.model.count_false_predicted) + "\n" \
                  + "Total test set accuracy is " + str(self.task.model.test_accuracy * 100.0) + " \n\n" \
                  + "Train accuracy is: " + str(self.task.model.train_accuracy * 100.0) + "% \n" \
                  + "Validation accuracy is: " + str(self.task.model.valid_accuracy * 100.0) + "%"
        self.ui.resultsTextEdit.setText(results)

    def set_buttons_handlers(self):
        self.ui.trainAgainBtn.clicked.connect(self.train_again_pressed)
        self.ui.saveBtn.clicked.connect(self.save_pressed)

    def train_again_pressed(self):
        self.close()
        # todo: consider to return to appropriate train window
        from gui_controllers.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()

    def save_pressed(self):
        self.close()
        from gui_controllers.saveModelWinController import SaveModelWinController
        self.window = SaveModelWinController(self.task)
        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainResultsWinController()
    MainWindow.show()
    sys.exit(app.exec_())
