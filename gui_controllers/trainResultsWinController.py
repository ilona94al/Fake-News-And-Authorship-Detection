from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox
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
        self.task.create_accuracy_graph()
        self.task.create_loss_graph()

        self.set_graph(widget=self.ui.accLabel, plot_path=self.task.acc_path)
        self.set_graph(widget=self.ui.lossLabel, plot_path=self.task.loss_path)

        results = "Number of true predicts: " + str(self.task.model.count_well_predicted) + "\n" \
                  + "Number of false predicts: " + str(self.task.model.count_false_predicted) + "\n" \
                  + "Total test set accuracy is " + "{:.1f}%".format(self.task.model.test_accuracy * 100.0) + "\n" \
                  + "Train accuracy is: " + "{:.1f}%".format(self.task.model.train_accuracy * 100.0) + "\n" \
                  + "Validation accuracy is: " + "{:.1f}%".format(self.task.model.valid_accuracy * 100.0)
        self.ui.resultTextEdit.setPlainText(results)

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.train_again_pressed)
        self.ui.saveBtn.clicked.connect(self.save_pressed)

    @staticmethod
    def set_graph(widget, plot_path):
        pixmap = QtGui.QPixmap("../" + plot_path)
        pixmap_small = pixmap.scaled(620, 420)
        widget.setPixmap(pixmap_small)

    def train_again_pressed(self):
        msg = QMessageBox()
        msg.setWindowTitle("Train again")
        msg.setText("Do you sure you want leave without saving?")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setInformativeText("The trained model will be lost")
        msg.buttonClicked.connect(self.pop_up_action)

        x = msg.exec_()

    def pop_up_action(self, i):
        if i.text() == 'OK':
            self.close()
            from model.plagiarism_task import PlagiarismTask
            if isinstance(self.task,PlagiarismTask):
                from gui_controllers.trainPlagModelWinController import TrainPlagModelWinController
                self.window = TrainPlagModelWinController()
            else:
                from gui_controllers.trainNewsModelWinController import TrainNewsModelWinController
                self.window = TrainNewsModelWinController()
            self.window.show()

    def save_pressed(self):
        self.close()
        from gui_controllers.saveModelWinController import SaveModelWinController
        self.window = SaveModelWinController(self.task)
        self.window.show()
