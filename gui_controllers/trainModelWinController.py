from PyQt5 import QtWidgets

from gui_controllers.formCheckerWinController import FormCheckerWinController


class TrainModelWinController(FormCheckerWinController):
    def __init__(self, parent=None):
        super(TrainModelWinController, self).__init__(parent)
        self.task = None

    def back_pressed(self):
        self.close()
        from gui_controllers.chooseTrainWinController import ChooseTrainWinController
        self.window = ChooseTrainWinController()
        self.window.show()


    def next(self):
        self.close()
        from gui_controllers.trainProgressWinController import TrainProgressWinController
        self.window = TrainProgressWinController(self.task)
        self.window.show()



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = TrainModelWinController()
    MainWindow.show()
    sys.exit(app.exec_())
