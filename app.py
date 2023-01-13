import sys

from PySide6.QtWidgets import QApplication

from librarys.interface import Window
from librarys.process import Identify
from librarys.client import Client
from qt_material import apply_stylesheet
import os

envpath = '~/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

class App:
    def __init__(self):
        super().__init__()
        self.qapp = QApplication(sys.argv)
        self.win = Window(self)
        self.win.show()
        self.identify = Identify(self.win)
        self.client = Client(self)

    def run(self, has_father_window=False):
        self.identify.start()
        self.client.start()
        if not has_father_window:
            sys.exit(self.qapp.exec_())


if __name__ == '__main__':
    app = App()
    app.run()