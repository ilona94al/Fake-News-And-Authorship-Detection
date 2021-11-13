# -*- coding: utf-8 -*-

# Form implementation generated from reading gui_design file 'save_model_window.gui_design'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SaveModelWindow(object):
    def setupUi(self, SaveModelWindow):
        SaveModelWindow.setObjectName("SaveModelWindow")
        SaveModelWindow.resize(860, 456)
        self.centralwidget = QtWidgets.QWidget(SaveModelWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundImg = QtWidgets.QLabel(self.centralwidget)
        self.backgroundImg.setGeometry(QtCore.QRect(0, 0, 860, 600))
        self.backgroundImg.setText("")
        self.backgroundImg.setPixmap(QtGui.QPixmap("../../RESOURCES/background.jpg"))
        self.backgroundImg.setScaledContents(True)
        self.backgroundImg.setObjectName("backgroundImg")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(30, 0, 800, 420))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.groupBox.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Sitka Heading")
        font.setPointSize(36)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet("color: rgb(255, 255, 255);")
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.saveBtn = QtWidgets.QPushButton(self.groupBox)
        self.saveBtn.setEnabled(True)
        self.saveBtn.setGeometry(QtCore.QRect(530, 329, 240, 60))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.saveBtn.setFont(font)
        self.saveBtn.setAcceptDrops(False)
        self.saveBtn.setAutoFillBackground(False)
        self.saveBtn.setStyleSheet("background-color: rgb(0, 111, 0);\n"
"\n"
"font: 75 16pt \"Sitka Small\";\n"
"color: rgb(255, 255, 255);\n"
"border-width: 3px;\n"
"border-radius: 30px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: solid;\n"
"\n"
"alternate-background-color: rgb(135, 135, 135);")
        self.saveBtn.setCheckable(False)
        self.saveBtn.setAutoDefault(False)
        self.saveBtn.setDefault(False)
        self.saveBtn.setFlat(False)
        self.saveBtn.setObjectName("saveBtn")
        self.backBtn = QtWidgets.QPushButton(self.groupBox)
        self.backBtn.setEnabled(True)
        self.backBtn.setGeometry(QtCore.QRect(30, 329, 240, 60))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.backBtn.setFont(font)
        self.backBtn.setAcceptDrops(False)
        self.backBtn.setAutoFillBackground(False)
        self.backBtn.setStyleSheet("\n"
"background-color: rgb(159, 0, 0);\n"
"font: 75 16pt \"Sitka Small\";\n"
"color: rgb(255, 255, 255);\n"
"border-width: 3px;\n"
"border-radius: 30px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: solid;")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:/My Documents/Studies/Semester_9/פרויקט מסכם - שלב ב/icons/pnghut_arrow-icon-direction-left-logo-text.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.backBtn.setIcon(icon)
        self.backBtn.setIconSize(QtCore.QSize(24, 24))
        self.backBtn.setCheckable(False)
        self.backBtn.setAutoDefault(False)
        self.backBtn.setDefault(False)
        self.backBtn.setFlat(False)
        self.backBtn.setObjectName("backBtn")
        self.verticalFrame = QtWidgets.QFrame(self.groupBox)
        self.verticalFrame.setGeometry(QtCore.QRect(100, 115, 600, 181))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.verticalFrame.setFont(font)
        self.verticalFrame.setAutoFillBackground(False)
        self.verticalFrame.setStyleSheet("border-width: 4px;\n"
"border-radius: 15px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"font: 10pt \"Sitka Small\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: qlineargradient(spread:pad, x1:0.996, y1:0.0340909, x2:1, y2:0, stop:1 rgba(0, 0, 32, 170));")
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout.setContentsMargins(18, 18, 18, 18)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalGroupBox = QtWidgets.QGroupBox(self.verticalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalGroupBox.sizePolicy().hasHeightForWidth())
        self.horizontalGroupBox.setSizePolicy(sizePolicy)
        self.horizontalGroupBox.setMinimumSize(QtCore.QSize(0, 0))
        self.horizontalGroupBox.setStyleSheet("")
        self.horizontalGroupBox.setObjectName("horizontalGroupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalGroupBox)
        self.horizontalLayout.setContentsMargins(10, 48, 10, 24)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.inputName = QtWidgets.QLineEdit(self.horizontalGroupBox)
        self.inputName.setEnabled(True)
        self.inputName.setStyleSheet("border-width: 2px;\n"
"border-radius: 5px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"background-color: rgb(181, 181, 181);\n"
"color: rgb(0, 0, 0);")
        self.inputName.setObjectName("inputName")
        self.horizontalLayout.addWidget(self.inputName)
        self.verticalLayout.addWidget(self.horizontalGroupBox)
        SaveModelWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SaveModelWindow)
        QtCore.QMetaObject.connectSlotsByName(SaveModelWindow)

    def retranslateUi(self, SaveModelWindow):
        _translate = QtCore.QCoreApplication.translate
        SaveModelWindow.setWindowTitle(_translate("SaveModelWindow", "Save Model Window"))
        self.groupBox.setTitle(_translate("SaveModelWindow", "Save a new model"))
        self.saveBtn.setText(_translate("SaveModelWindow", "Save"))
        self.backBtn.setText(_translate("SaveModelWindow", "Back"))
        self.horizontalGroupBox.setTitle(_translate("SaveModelWindow", "Give a name for your model :"))
        self.inputName.setPlaceholderText(_translate("SaveModelWindow", "example: fake news model 8-11-2021"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SaveModelWindow = QtWidgets.QMainWindow()
    ui = Ui_SaveModelWindow()
    ui.setupUi(SaveModelWindow)
    SaveModelWindow.show()
    sys.exit(app.exec_())