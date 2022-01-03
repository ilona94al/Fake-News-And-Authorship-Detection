from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from gui_controllers.formCheckerWinController import FormCheckerWinController


class SaveModelWinController(FormCheckerWinController):
    def __init__(self, task, parent=None):
        super(SaveModelWinController, self).__init__(parent)
        self.task = task
        from gui_design.save_model_window import Ui_SaveModelWindow
        self.ui = Ui_SaveModelWindow()
        self.ui.setupUi(self)

        from model.plagiarism_task import PlagiarismTask
        if isinstance(task, PlagiarismTask):
            self.ui.inputName.setText(task.author)
        else:
            self.ui.inputName.setPlaceholderText("for example: Fake news 1")
        self.set_buttons_handlers()
        self.clear_feedback()

    def set_buttons_handlers(self):
        self.ui.backBtn.clicked.connect(self.back_pressed)
        self.ui.saveBtn.clicked.connect(self.save_pressed)

    def back_pressed(self):
        self.close()
        from gui_controllers.trainResultsWinController import TrainResultsWinController
        self.window = TrainResultsWinController(self.task)
        self.window.show()

    def save_pressed(self):
        name_widget = self.ui.inputName
        self.name_widget_str = name_widget.text()

        if self.name_widget_str == "":
            self.invalid_input("No name, please give a name for your model", name_widget)
        else:
            self.set_normal_style(name_widget)
            if self.task.save_model(self.name_widget_str):
                self.back_to_main_win()
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Save model")
                msg.setText("Do you want replace model with name '" + self.name_widget_str + "' ?")
                msg.setIcon(QMessageBox.Warning)
                msg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
                msg.setDefaultButton(QMessageBox.No)
                msg.setInformativeText("There is already a model with the same name in this location.")
                msg.buttonClicked.connect(self.pop_up_action)
                x = msg.exec_()

    def pop_up_action(self, i):
        if i.text() == '&Yes':
            if self.task.save_model(model_name=self.name_widget_str, replace=True):
                self.back_to_main_win()
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Save model")
                msg.setText("Error occur during save model process")
                msg.setIcon(QMessageBox.Warning)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setDefaultButton(QMessageBox.Ok)
                msg.setInformativeText("Try Again")
                msg.buttonClicked.connect(lambda i: None)
                x = msg.exec_()

    def back_to_main_win(self):
        self.close()
        from gui_controllers.mainWinController import MainWinController
        self.window = MainWinController()
        self.window.show()
