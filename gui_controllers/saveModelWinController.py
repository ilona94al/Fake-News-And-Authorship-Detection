from PyQt5 import QtWidgets

from gui_controllers.formCheckerWinController import FormCheckerWinController


class SaveModelWinController(FormCheckerWinController):
    def __init__(self, task, parent=None):
        super(SaveModelWinController, self).__init__(parent)
        self.task = task
        from gui_design.save_model_window import Ui_SaveModelWindow
        self.ui = Ui_SaveModelWindow()
        self.ui.setupUi(self)
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
        name_widget_str=name_widget.text()

        if name_widget_str=="":
            self.invalid_input("No name, please give a name for your model",name_widget)
        else:
            self.set_normal_style(name_widget)
            self.task.save_model(name_widget_str)
            self.close()
            from gui_controllers.mainWinController import MainWinController
            self.window = MainWinController()
            self.window.show()



