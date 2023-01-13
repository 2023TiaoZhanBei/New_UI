# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////
import time
from re import S
import sys
import os
import platform

from PySide6.QtCore import Signal

from librarys.reaction import Reaction
# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
from threading import Event
from librarys.process import Identify
from librarys.client import Client

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////


class MainWindow(QMainWindow):
    received = Signal(str)
    def __init__(self, app):
        QMainWindow.__init__(self)
        
        self.app = app

        # SET AS GLOBAL WIDGETS
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        title = "PyDracula - Modern GUI"
        description = "PyDracula APP - Theme with colors based on Dracula for Python."
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        UIFunctions.uiDefinitions(self)

        # QTableWidget PARAMETERS
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # BUTTONS CLICK

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)

        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)

        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        self.show()

        # SET CUSTOM THEME
        useCustomTheme = True
        themeFile = "themes/py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))
        
        # Init Values
        self.init_camera_values()

    def closeEvent(self, a0) -> None:
        self.app.client.stop_ping()

    def join(self):
        # self.app.client.send(self.app.client.type)
        self.name = self.ui.lineEdit_2.text()
        if self.name != '':
            self.app.client.send(self.name)  # 发送用户名
            widgets.btn_join.setEnabled(False)
            widgets.lineEdit.setEnabled(False)
            widgets.btn_join.setText("已加入会议")
            self.isLogin = True
            self.app.identify.start()
        else:
            self.show_error("请输入用户名")
            widgets.lineEdit.setFocus()


    def show_error(self, msg: str):
        QMessageBox.information(self, "错误", msg)
        
    def init_camera_values(self):
        widgets.btn_start.clicked.connect(self.switch_camera_status)
        self.eventRunning = Event()

        self.member_list = []

        self.received.connect(self.get_data)
        self.isLogin = False
        self.isTarget = False
        self.isController = False
        self.isControlling = False
        # self.btn_get_ctrl.clicked.connect(self.get_ctrl)
        self.type = 'receiver'
        self.name = ''
        self.target = ''
        widgets.btn_join.clicked.connect(self.join)
        # self.btn_pause.clicked.connect(self.switch_ctrl)
        # self.checkBox.stateChanged.connect(self.switch_target)

        self.reaction = Reaction()
        # self.action_tutorials.triggered.connect(self.show_tutorials_win)
        # self.action_help.triggered.connect(self.show_help_win)

    def get_data(self, data: str):
        if data == 'pong':
            return
        elif data == 'ping':
            self.app.client.timer.start()
            return
        splits = data.split(' ')
        if splits[0] == 'command':
            self.set_log("控制者发出指令：" + splits[1])
            if self.isTarget:
                # 响应指令：pyuserinput
                self.reaction.react(splits[1])

        elif splits[0] == 'change_controller':
            self.isControlling = False
            widgets.label_controller.setText(("正在控制：" if self.isControlling else "控制已暂停：")
                                          + splits[1])
            if splits[1] == self.name:
                self.isController = True
                widgets.btn_get_ctrl.setText("退出控制")
            else:
                self.isController = False
                widgets.btn_get_ctrl.setText("获取控制")
        elif splits[0] == 'control_switched':
            self.isControlling = not self.isControlling
            widgets.label_controller.setText(("正在控制：" if self.isControlling else "控制已暂停：")
                                          + splits[1])
        elif splits[0] == 'member_list':
            self.member_list = splits[1:]
            self.init_list_view()
        elif splits[0] == 'duplicate_name':
            self.show_error("用户名已存在，请修改")
            widgets.lineEdit.setEnabled(True)
            widgets.lineEdit.setFocus()
            widgets.btn_join.setEnabled(True)
            widgets.btn_join.setText("加入会议")
            self.isLogin = False

    def init_list_view(self):
        slm = QStringListModel()  # 创建model
        slm.setStringList(self.member_list)  # 将数据设置到model
        widgets.listView.setModel(slm)  # 绑定 listView 和 model
        widgets.listView.clicked.connect(self.clicked_list)  # listview 的点击事件

    def clicked_list(self, q_model_index):
        self.target = self.member_list[q_model_index.row()]
    def set_log(self, msg):
        widgets.textBrowser.append(time.strftime("(%H:%M:%S)", time.localtime()) + ' ' + msg)
        widgets.textBrowser.moveCursor(widgets.textBrowser.textCursor().End)
    def switch_camera_status(self):
        if self.eventRunning.isSet():
            widgets.label_img.setText("Hello\nWorld")
            widgets.btn_start.setText("开启识别")
            self.eventRunning.clear()
        else:
            self.eventRunning.set()
            widgets.btn_start.setText("停止识别")
    
    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW NEW PAGE
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.mainPage)  # SET PAGE
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU
            widgets.btn_start.clicked.connect(self.switch)
            

        if btnName == "btn_save":
            print("Save BTN clicked!")

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')

    # RESIZE EVENTS
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')


class App:
    def __init__(self):
        super().__init__()
        self.qapp = QApplication(sys.argv)
        self.win = MainWindow(self)
        self.win.show()
        self.identify = Identify(self.win)
        self.client = Client(self)

    def run(self, has_father_window=False):
        self.identify.start()
        self.client.start()
        if not has_father_window:
            sys.exit(self.qapp.exec_())


if __name__ == "__main__":
    app = App()
    app.run()
