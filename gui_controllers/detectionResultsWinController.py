
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui, QtWidgets


class DetectionResultsWinController(QMainWindow):
    def __init__(self, detection, parent=None):
        super(DetectionResultsWinController, self).__init__(parent)
        from gui_design.detection_results_window import Ui_DetectionResultsWindow
        self.ui = Ui_DetectionResultsWindow()
        self.ui.setupUi(self)

        self.detection=detection

        self.update_ui_with_data(detection)

        self.set_buttons_handlers()

    def update_ui_with_data(self, detection):
        detection.create_probabilities_plot()
        detection.create_distribution_plot()

        self.set_graph(widget=self.ui.plotLabel1,plot_path=detection.plot_path1)
        self.set_graph(widget=self.ui.plotLabel2,plot_path=detection.plot_path2)

        text=detection.get_result()

        self.ui.detectionTextEdit.setPlainText(text)

    def set_graph(self, widget, plot_path):
        pixmap = QtGui.QPixmap("../" + plot_path)
        pixmap_small = pixmap.scaled(620, 420)
        widget.setPixmap(pixmap_small)

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.back_pressed)

    def back_pressed(self):
        self.close()
        from model.fake_news_detection import FakeNewsDetection
        if isinstance(self.detection, FakeNewsDetection):
            from gui_controllers.fakeNewsWinController import FakeNewsWinController
            self.window = FakeNewsWinController()
        from model.plagiarism_detection import PlagiarismDetection
        if isinstance(self.detection, PlagiarismDetection):
            from gui_controllers.plagiarismWinController import PlagiarismWinController
            self.window = PlagiarismWinController()
        self.window.show()





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = DetectionResultsWinController()
    MainWindow.show()
    sys.exit(app.exec_())

