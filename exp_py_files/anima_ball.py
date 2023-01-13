import sys

from PySide6.QtGui import Qt
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QPropertyAnimation, QPoint, QEasingCurve
from PySide6.QtWidgets import QApplication


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(600, 600)
        self.child = QWidget(self)
        self.child.setStyleSheet("background-color:red;border-radius:15px;")
        self.child.resize(30, 30)
        self.anim = QPropertyAnimation(self.child, b"pos")
        self.anim.setEasingCurve(QEasingCurve.Type.InBounce)
        self.anim.setEndValue(QPoint(400, 400))
        self.anim.setDuration(1500)
        self.anim.start()

if __name__ == "__main__":

    app = QApplication([])
    win = Window()
    # 透明
    win.setWindowFlags(win.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    win.setAttribute(Qt.WA_TranslucentBackground)
    win.show()
    sys.exit(app.exec())
