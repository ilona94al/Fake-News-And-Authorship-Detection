
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui, QtWidgets


class DetectionResultsWinController(QMainWindow):
    def __init__(self, detection, parent=None):
        super(DetectionResultsWinController, self).__init__(parent)

        from gui_design.detection_results_window import Ui_DetectionResultsWindow
        self.ui = Ui_DetectionResultsWindow()
        self.ui.setupUi(self)

        self.update_ui_with_data(detection)

        self.set_buttons_handlers()

    def update_ui_with_data(self, detection):
        detection.create_probabilities_plot()
        detection.create_distribution_plot()

        self.set_graph(widget=self.ui.plotLabel1,plot_path=detection.plot_path1)
        self.set_graph(widget=self.ui.plotLabel2,plot_path=detection.plot_path2)

        text = "The book: \"" + str(detection.book_name) + \
               "\" was written by " + str(detection.author_name) \
               + " with " + "{:.1f}%".format(detection.real_percent) + " certainty." + "\n"
        if detection.real_percent>detection.fake_percent:
            text+="It seems like the book was written by "+detection.author_name+"."
        else:
            text+="It seems like the book wasn't written by Shakespeare"
        self.ui.detectionTextEdit.setPlainText(text)

    def set_graph(self, widget, plot_path):
        pixmap = QtGui.QPixmap("../" + plot_path)
        pixmap_small = pixmap.scaled(620, 420)
        widget.setPixmap(pixmap_small)

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.back_pressed)

    def back_pressed(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = DetectionResultsWinController()
    MainWindow.show()
    sys.exit(app.exec_())

