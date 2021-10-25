import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow


class FakeNewsWinController(QMainWindow):
    def __init__(self, parent=None):
        super(FakeNewsWinController, self).__init__(parent)
        from gui.fake_news_window import Ui_FakeNewsWindow

        self.ui = Ui_FakeNewsWindow()
        self.ui.setupUi(self)

        self.ui.backBtn.clicked.connect(self.back_pressed)

        self.ui.startBtn.clicked.connect(self.start_pressed)

        self.ui.errorMsg.setHidden(True)

        self.ui.trainedModelsComboBox.clear()
        os.chdir("../Model/")
        arr = os.listdir('FakeNews')
        self.ui.trainedModelsComboBox.addItems(arr)
        #self.ui.trainedModelsComboBox.addItem("tm")

    def back_pressed(self):
        self.close()
        from gui.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()



    def start_pressed(self):
        self.clear_feedback()

        tweet=str(self.ui.inputTweet.toPlainText());
        model_name = self.ui.trainedModelsComboBox.currentText()
        if tweet == "":
            self.empty_tweet()
        else:
            self.close()
            from gui.detectionResultsWinController import DetectionResultsWinController
            # todo: results= DETECT(model_name, tweet)
            #  find the relevant trained model(according to the model name)
            #  insert tweet as input to the model and get detection results (graphs and etc.)
            self.window = DetectionResultsWinController()#input results
            self.window.show()

    def clear_feedback(self):
        self.ui.errorMsg.setHidden(True)
        self.ui.errorMsg.setText("")
        self.ui.horizontalGroupBox.setStyleSheet("")

    def empty_tweet(self):
        msg = "Empty tweet box!\n please paste a tweet"
        widget = self.ui.horizontalGroupBox
        self.setFeedback(msg, widget)


    def setFeedback(self, msg, widget):
        self.ui.errorMsg.setText(msg)
        self.ui.errorMsg.adjustSize()
        self.ui.errorMsg.setHidden(False)
        widget.setStyleSheet("border-color: rgb(170, 0, 0);")
        widget.update()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = FakeNewsWinController()
    MainWindow.show()
    sys.exit(app.exec_())
