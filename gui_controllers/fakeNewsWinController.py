import os

from PyQt5 import QtWidgets

from gui_controllers.formCheckerWinController import FormCheckerWinController


class FakeNewsWinController(FormCheckerWinController):
    def __init__(self, parent=None):
        super(FakeNewsWinController, self).__init__(parent)
        from gui_design.fake_news_window import Ui_FakeNewsWindow

        self.ui = Ui_FakeNewsWindow()
        self.ui.setupUi(self)

        self.set_buttons_handlers()

        self.update_ui_with_data()

        self.clear_feedback()

    def update_ui_with_data(self):
        self.ui.trainedModelsComboBox.clear()
        from constants import TRAINED_MODELS_PATH
        arr = os.listdir('../' + TRAINED_MODELS_PATH + 'FakeNews')
        models = []
        for item in arr:
            if item.split(".")[1] == "h5":
                models.append(item)
        self.ui.trainedModelsComboBox.addItems(models)

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.startBtn.clicked.connect(self.start_pressed)

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()



    def start_pressed(self):
        self.clear_feedback()

        tweet_widget = self.ui.inputTweet

        self.set_normal_style(tweet_widget)

        tweet_str=str(tweet_widget.toPlainText());

        model_name = self.ui.trainedModelsComboBox.currentText()

        if tweet_str == "":
            self.invalid_input( "Empty tweet box!\n please paste a tweet",tweet_widget)
        else:
            self.set_normal_style(tweet_widget)

            self.close()
          # todo: results= DETECT(model_name, tweet)
            #  find the relevant trained model(according to the model name)
            #  insert tweet as input to the model and get detection results (graphs and etc.)





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = FakeNewsWinController()
    MainWindow.show()
    sys.exit(app.exec_())
