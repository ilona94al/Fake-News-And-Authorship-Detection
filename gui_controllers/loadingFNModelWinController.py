import threading

from gui_controllers.loadingWinController import LoadingWinController


class LoadingFNModelWinController(LoadingWinController):
    def __init__(self,  content, model_name, parent=None):
        super(LoadingFNModelWinController, self).__init__(parent)

        #   thread that loads the model, runs the model with input from user
        self.t1 = threading.Thread(target=self.run_detection, args=(content, model_name))
        self.t1.start()

        #   thread that waiting until first thread will finish
        #   and then raises event for the next step
        self.t2 = threading.Thread(target=self.run_waiting)
        self.t2.start()


    def run_detection(self, content, model_name):
        from model.fake_news_detection import FakeNewsDetection
        self.detection = FakeNewsDetection(input=content, model_name=model_name + ".h5")
        self.loading = False





