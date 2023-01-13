import sys

from PySide6.QtGui import Qt
from PySide6.QtWidgets import QWidget, QDialog
from PySide6.QtCore import QPropertyAnimation, QPoint, QEasingCurve
from PySide6.QtWidgets import QApplication


class FBall(QDialog):
    def __init__(self, size, color = 'red'):
        super(FBall, self).__init__()

        self.ball_weight = QWidget(self)
        self.ball_weight.resize(size, size)
        self.ball_weight.setStyleSheet(f'background-color: {color}; border-radius: {size//2}px;')

        self.setWindowFlags(Qt.FramelessWindowHint)

        self.setAttribute(Qt.WA_TranslucentBackground)

        self.anim = None

    def expand(self):
        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.setEndValue(QPoint(1000, 1000))
        self.anim.setDuration(1500)
        self.anim.start()


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.mainland_ball = FBall(100, 'red')



        self.mainland_ball.show()

        self.mainland_ball.expand()



if __name__ == "__main__":

    app = QApplication([])
    win = Window()
    # 透明
    # win.setAttribute(Qt.WA_TranslucentBackground)
    # win.show()
    sys.exit(app.exec())
