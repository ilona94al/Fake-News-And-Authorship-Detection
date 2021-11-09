from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow

combo_box_options = {
    'option1': "Plagiarism",
    'option2': "Fake news"
}


class ChooseTrainWinController(QMainWindow):
    def __init__(self, parent=None):
        super(ChooseTrainWinController, self).__init__(parent)
        from gui_design.choose_train_window import Ui_ChooseTrainWindow

        self.ui = Ui_ChooseTrainWindow()
        self.ui.setupUi(self)

        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.nextBtn.clicked.connect(self.next_pressed)

        self.ui.modelTypeComboBox.addItem(combo_box_options['option1'])
        self.ui.modelTypeComboBox.addItem(combo_box_options['option2'])

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()

    def next_pressed(self):

        model_type = self.ui.modelTypeComboBox.currentText()
        self.close()

        if model_type == combo_box_options['option1']:
            from gui_controllers.trainPlagModelWinController import TrainPlagModelWinController
            self.window = TrainPlagModelWinController()
        if model_type == combo_box_options['option2']:
            from gui_controllers.trainNewsModelWinController import TrainNewsModelWinController
            self.window = TrainNewsModelWinController()

        self.window.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ChooseTrainWinController()
    MainWindow.show()
    sys.exit(app.exec_())
