# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'choose_train_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ChooseTrainWindow(object):
    def setupUi(self, ChooseTrainWindow):
        ChooseTrainWindow.setObjectName("ChooseTrainWindow")
        ChooseTrainWindow.resize(859, 398)
        self.centralwidget = QtWidgets.QWidget(ChooseTrainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundImg = QtWidgets.QLabel(self.centralwidget)
        self.backgroundImg.setGeometry(QtCore.QRect(0, 0, 860, 600))
        self.backgroundImg.setText("")
        self.backgroundImg.setPixmap(QtGui.QPixmap("resources/background.jpg"))
        self.backgroundImg.setScaledContents(True)
        self.backgroundImg.setObjectName("backgroundImg")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(30, 0, 800, 370))
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
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
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
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.verticalGroupBox = QtWidgets.QGroupBox(self.groupBox)
        self.verticalGroupBox.setGeometry(QtCore.QRect(150, 115, 500, 121))
        self.verticalGroupBox.setStyleSheet("border-width: 4px;\n"
"border-radius: 15px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"font: 12pt \"Sitka Small\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: qlineargradient(spread:pad, x1:0.996, y1:0.0340909, x2:1, y2:0, stop:1 rgba(0, 0, 32, 170));")
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.verticalLayout_2.setContentsMargins(10, 48, 10, 24)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.modelTypeComboBox = QtWidgets.QComboBox(self.verticalGroupBox)
        self.modelTypeComboBox.setStyleSheet("border-width: 2px;\n"
"border-radius: 5px;\n"
"background-color: rgb(195, 195, 195);\n"
"border-style: solid;\n"
"color: rgb(0, 0, 0);")
        self.modelTypeComboBox.setObjectName("modelTypeComboBox")
        self.verticalLayout_2.addWidget(self.modelTypeComboBox)
        self.nextBtn = QtWidgets.QPushButton(self.groupBox)
        self.nextBtn.setEnabled(True)
        self.nextBtn.setGeometry(QtCore.QRect(530, 279, 240, 60))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.nextBtn.setFont(font)
        self.nextBtn.setAcceptDrops(False)
        self.nextBtn.setAutoFillBackground(False)
        self.nextBtn.setStyleSheet("background-color: rgb(0, 111, 0);\n"
"\n"
"font: 75 16pt \"Sitka Small\";\n"
"color: rgb(255, 255, 255);\n"
"border-width: 3px;\n"
"border-radius: 30px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: solid;\n"
"\n"
"alternate-background-color: rgb(135, 135, 135);")
        self.nextBtn.setCheckable(False)
        self.nextBtn.setAutoDefault(False)
        self.nextBtn.setDefault(False)
        self.nextBtn.setFlat(False)
        self.nextBtn.setObjectName("nextBtn")
        self.backBtn = QtWidgets.QPushButton(self.groupBox)
        self.backBtn.setEnabled(True)
        self.backBtn.setGeometry(QtCore.QRect(30, 279, 240, 60))
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
        ChooseTrainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(ChooseTrainWindow)
        QtCore.QMetaObject.connectSlotsByName(ChooseTrainWindow)

    def retranslateUi(self, ChooseTrainWindow):
        _translate = QtCore.QCoreApplication.translate
        ChooseTrainWindow.setWindowTitle(_translate("ChooseTrainWindow", "Fake Texts Detector"))
        self.groupBox.setTitle(_translate("ChooseTrainWindow", "Train Model"))
        self.verticalGroupBox.setTitle(_translate("ChooseTrainWindow", "Please, choose model type for train:"))
        self.nextBtn.setText(_translate("ChooseTrainWindow", "Next"))
        self.backBtn.setText(_translate("ChooseTrainWindow", "Back"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ChooseTrainWindow = QtWidgets.QMainWindow()
    ui = Ui_ChooseTrainWindow()
    ui.setupUi(ChooseTrainWindow)
    ChooseTrainWindow.show()
    sys.exit(app.exec_())
