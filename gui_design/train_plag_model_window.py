# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'train_plag_model_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TrainPlagModelWindow(object):
    def setupUi(self, TrainPlagModelWindow):
        TrainPlagModelWindow.setObjectName("TrainPlagModelWindow")
        TrainPlagModelWindow.resize(860, 598)
        self.centralwidget = QtWidgets.QWidget(TrainPlagModelWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.backgroundImg = QtWidgets.QLabel(self.centralwidget)
        self.backgroundImg.setGeometry(QtCore.QRect(0, 0, 860, 600))
        self.backgroundImg.setText("")
        self.backgroundImg.setPixmap(QtGui.QPixmap("../RESOURCES/background.jpg"))
        self.backgroundImg.setScaledContents(True)
        self.backgroundImg.setObjectName("backgroundImg")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(30, 0, 800, 570))
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
        self.groupBox.setStyleSheet("color: rgb(255, 255, 255);\n"
"                    ")
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.trainBtn = QtWidgets.QPushButton(self.groupBox)
        self.trainBtn.setEnabled(True)
        self.trainBtn.setGeometry(QtCore.QRect(530, 479, 240, 60))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.trainBtn.setFont(font)
        self.trainBtn.setAcceptDrops(False)
        self.trainBtn.setAutoFillBackground(False)
        self.trainBtn.setStyleSheet("\n"
"                            background-color: rgb(0, 0, 20);\n"
"                            font: 75 16pt \"Sitka Small\";\n"
"                            color: rgb(255, 255, 255);\n"
"                            border-width: 3px;\n"
"                            border-radius: 30px;\n"
"                            border-color: rgb(255, 255, 255);\n"
"                            border-style: solid;\n"
"                        ")
        self.trainBtn.setCheckable(False)
        self.trainBtn.setAutoDefault(False)
        self.trainBtn.setDefault(False)
        self.trainBtn.setFlat(False)
        self.trainBtn.setObjectName("trainBtn")
        self.backBtn = QtWidgets.QPushButton(self.groupBox)
        self.backBtn.setEnabled(True)
        self.backBtn.setGeometry(QtCore.QRect(30, 479, 240, 60))
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
"                            background-color: rgb(159, 0, 0);\n"
"                            font: 75 16pt \"Sitka Small\";\n"
"                            color: rgb(255, 255, 255);\n"
"                            border-width: 3px;\n"
"                            border-radius: 30px;\n"
"                            border-color: rgb(255, 255, 255);\n"
"                            border-style: solid;\n"
"                        ")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../RESOURCES/left_arrow_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.backBtn.setIcon(icon)
        self.backBtn.setIconSize(QtCore.QSize(24, 24))
        self.backBtn.setCheckable(False)
        self.backBtn.setAutoDefault(False)
        self.backBtn.setDefault(False)
        self.backBtn.setFlat(False)
        self.backBtn.setObjectName("backBtn")
        self.verticalFrame1 = QtWidgets.QFrame(self.groupBox)
        self.verticalFrame1.setGeometry(QtCore.QRect(140, 130, 520, 291))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.verticalFrame1.setFont(font)
        self.verticalFrame1.setAutoFillBackground(False)
        self.verticalFrame1.setStyleSheet("border-width: 3px;\n"
"                            border-radius: 15px;\n"
"                            border-color: rgb(0, 0, 0);\n"
"                            border-style: solid;\n"
"                            font: 8pt \"Sitka Small\";\n"
"                            color: rgb(255, 255, 255);\n"
"                            background-color: qlineargradient(spread:pad, x1:0.996, y1:0.0340909, x2:1, y2:0, stop:1\n"
"                            rgba(0, 0, 32, 170));\n"
"                        ")
        self.verticalFrame1.setObjectName("verticalFrame1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame1)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalGroupBox11 = QtWidgets.QGroupBox(self.verticalFrame1)
        self.verticalGroupBox11.setStyleSheet("border-width: 2px;\n"
"font: 10pt \"Sitka Small\";")
        self.verticalGroupBox11.setObjectName("verticalGroupBox11")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalGroupBox11)
        self.verticalLayout_6.setContentsMargins(5, 36, 5, 18)
        self.verticalLayout_6.setSpacing(2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.inputAuthorName111 = QtWidgets.QLineEdit(self.verticalGroupBox11)
        self.inputAuthorName111.setStyleSheet("border-width: 3px;\n"
"border-radius: 5px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"background-color: rgb(188, 188, 188);\n"
"color: rgb(0, 0, 0);\n"
"                                                \n"
"")
        self.inputAuthorName111.setObjectName("inputAuthorName111")
        self.verticalLayout_6.addWidget(self.inputAuthorName111)
        self.label112 = QtWidgets.QLabel(self.verticalGroupBox11)
        self.label112.setStyleSheet("border-width: 0px;\n"
"")
        self.label112.setObjectName("label112")
        self.verticalLayout_6.addWidget(self.label112)
        self.horizontalLayout113 = QtWidgets.QHBoxLayout()
        self.horizontalLayout113.setObjectName("horizontalLayout113")
        self.inputPath1131 = QtWidgets.QLineEdit(self.verticalGroupBox11)
        self.inputPath1131.setStyleSheet("border-width: 3px;\n"
"border-radius: 5px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"background-color: rgb(188, 188, 188);\n"
"color: rgb(0, 0, 0);\n"
"                                                \n"
"")
        self.inputPath1131.setReadOnly(True)
        self.inputPath1131.setObjectName("inputPath1131")
        self.horizontalLayout113.addWidget(self.inputPath1131)
        self.uploadBtn1132 = QtWidgets.QToolButton(self.verticalGroupBox11)
        self.uploadBtn1132.setMinimumSize(QtCore.QSize(0, 0))
        self.uploadBtn1132.setMouseTracking(False)
        self.uploadBtn1132.setAutoFillBackground(False)
        self.uploadBtn1132.setStyleSheet("background-color: rgb(212, 212, 212);\n"
"                                                            border-width: 3px;\n"
"                                                            border-radius: 5px;\n"
"                                                            border-color: rgb(0, 0, 0);\n"
"                                                            border-style: solid;\n"
"                                                        ")
        self.uploadBtn1132.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/upload_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.uploadBtn1132.setIcon(icon1)
        self.uploadBtn1132.setObjectName("uploadBtn1132")
        self.horizontalLayout113.addWidget(self.uploadBtn1132)
        self.verticalLayout_6.addLayout(self.horizontalLayout113)
        self.label114 = QtWidgets.QLabel(self.verticalGroupBox11)
        self.label114.setStyleSheet("border-width: 0px;\n"
"")
        self.label114.setObjectName("label114")
        self.verticalLayout_6.addWidget(self.label114)
        self.horizontalLayout115 = QtWidgets.QHBoxLayout()
        self.horizontalLayout115.setObjectName("horizontalLayout115")
        self.label1151 = QtWidgets.QLabel(self.verticalGroupBox11)
        self.label1151.setStyleSheet("border-width: 0px;\n"
"")
        self.label1151.setObjectName("label1151")
        self.horizontalLayout115.addWidget(self.label1151)
        self.inputEpoch1152 = QtWidgets.QLineEdit(self.verticalGroupBox11)
        self.inputEpoch1152.setStyleSheet("border-width: 3px;\n"
"border-radius: 5px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"background-color: rgb(188, 188, 188);\n"
"color: rgb(0, 0, 0);\n"
"                                                \n"
"")
        self.inputEpoch1152.setObjectName("inputEpoch1152")
        self.horizontalLayout115.addWidget(self.inputEpoch1152)
        self.label1153 = QtWidgets.QLabel(self.verticalGroupBox11)
        self.label1153.setStyleSheet("border-width: 0px;\n"
"")
        self.label1153.setObjectName("label1153")
        self.horizontalLayout115.addWidget(self.label1153)
        self.inputBatchSz1154 = QtWidgets.QLineEdit(self.verticalGroupBox11)
        self.inputBatchSz1154.setStyleSheet("border-width: 3px;\n"
"border-radius: 5px;\n"
"border-color: rgb(0, 0, 0);\n"
"border-style: solid;\n"
"background-color: rgb(188, 188, 188);\n"
"color: rgb(0, 0, 0);\n"
"                                                \n"
"")
        self.inputBatchSz1154.setObjectName("inputBatchSz1154")
        self.horizontalLayout115.addWidget(self.inputBatchSz1154)
        self.verticalLayout_6.addLayout(self.horizontalLayout115)
        self.verticalLayout.addWidget(self.verticalGroupBox11)
        self.errorMsg = QtWidgets.QLabel(self.groupBox)
        self.errorMsg.setGeometry(QtCore.QRect(280, 440, 80, 41))
        self.errorMsg.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.errorMsg.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.983, y1:0.00568182,\n"
"                            x2:0.996, y2:0, stop:1 rgba(48, 48, 48, 143));\n"
"                            color: rgb(170, 0, 0);\n"
"                            font: 75 8pt \"System\";\n"
"                            border-radius: 10px;\n"
"                            border-style: solid;\n"
"                            border-width: 0px\n"
"                        ")
        self.errorMsg.setObjectName("errorMsg")
        TrainPlagModelWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(TrainPlagModelWindow)
        QtCore.QMetaObject.connectSlotsByName(TrainPlagModelWindow)

    def retranslateUi(self, TrainPlagModelWindow):
        _translate = QtCore.QCoreApplication.translate
        TrainPlagModelWindow.setWindowTitle(_translate("TrainPlagModelWindow", "Train New Plagiarism Model"))
        self.groupBox.setTitle(_translate("TrainPlagModelWindow", "Train a new Model"))
        self.trainBtn.setText(_translate("TrainPlagModelWindow", "Train"))
        self.backBtn.setText(_translate("TrainPlagModelWindow", "Back"))
        self.verticalGroupBox11.setTitle(_translate("TrainPlagModelWindow", "Author name:"))
        self.inputAuthorName111.setPlaceholderText(_translate("TrainPlagModelWindow", "William Shakespeare"))
        self.label112.setText(_translate("TrainPlagModelWindow", "Path to directory with books (.txt files):"))
        self.inputPath1131.setPlaceholderText(_translate("TrainPlagModelWindow", "/.../..."))
        self.label114.setText(_translate("TrainPlagModelWindow", "Training Hyperparameters:"))
        self.label1151.setText(_translate("TrainPlagModelWindow", "Epoch:"))
        self.inputEpoch1152.setPlaceholderText(_translate("TrainPlagModelWindow", "15"))
        self.label1153.setText(_translate("TrainPlagModelWindow", "Batch size"))
        self.inputBatchSz1154.setPlaceholderText(_translate("TrainPlagModelWindow", "10"))
        self.errorMsg.setText(_translate("TrainPlagModelWindow", "warning"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TrainPlagModelWindow = QtWidgets.QMainWindow()
    ui = Ui_TrainPlagModelWindow()
    ui.setupUi(TrainPlagModelWindow)
    TrainPlagModelWindow.show()
    sys.exit(app.exec_())
