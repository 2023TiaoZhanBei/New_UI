import math
import sys
import time
import warnings
from dataclasses import dataclass

import PySide6
from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QApplication, QDialog, QWidget

warnings.filterwarnings("ignore")


class FBall(QDialog):
    def __init__(self, size, color="red"):
        super(FBall, self).__init__()
        self.iniDragCor = [0, 0]
        self.ball_size = size
        self.ball_weight = QWidget(self)
        self.ball_weight.resize(size, size)

        self.ball_weight.setStyleSheet(
            f"background-color: {color}; border-radius: {size // 2}px; "
        )

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # 设置不能调整大小
        self.setFixedSize(size, size)

        self.setAttribute(Qt.WA_TranslucentBackground)

        self.anim = None

    def expand(self, offset_x, offset_y):
        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        # 获取当前位置
        self.anim.setEndValue(
            QPoint(offset_x - (self.ball_size // 2), offset_y - (self.ball_size // 2))
        )
        self.anim.setDuration(800)
        self.anim.start()

    def shrink(self, offset_x, offset_y):
        self.expand(offset_x, offset_y)

    def update_position(self, e):
        if e.button() == Qt.LeftButton:
            self.iniDragCor[0] = e.x()
            self.iniDragCor[1] = e.y()

    def move_related(self, e):
        x = e.x() - self.iniDragCor[0]
        y = e.y() - self.iniDragCor[1]
        cor = QPoint(x, y)
        self.move(self.mapToParent(cor))

    def enterEvent(self, event: PySide6.QtGui.QEnterEvent) -> None:
        print("enter")
        return super().enterEvent(event)

    def leaveEvent(self, event: PySide6.QtCore.QEvent) -> None:
        print("leave")
        return super().leaveEvent(event)


@dataclass
class MainLandBall(FBall):
    color_set = [
        (83, 109, 254),
        (124, 77, 255),
        (255, 64, 129),
        (255, 82, 82),
        (83, 109, 254),
    ]
    color_set_deep = [
        (197, 202, 233),
        (209, 196, 233),
        (225, 190, 231),
        (248, 187, 208),
        (255, 205, 210),
    ]
    offset = 180

    def __init__(self, size, color="red", child_balls_num=5):
        super(MainLandBall, self).__init__(size, color)

        self.balls = []
        self.is_expanded = False
        self.move_start_time = 0
        # 创建多个小球
        for i in range(child_balls_num):
            self.balls.append(FBall(size // 2))

    def expand(self, move_length, pre_angle=40):

        x0, y0 = self.pos().x() + (self.ball_size // 2), self.pos().y() + (
            self.ball_size // 2
        )
        self.is_expanded = True

        for i, (angle_i, color, color_deep) in enumerate(
            zip(
                range(0 + self.offset, int(pre_angle * 5 + 1 + self.offset), pre_angle),
                self.color_set,
                self.color_set_deep,
            )
        ):
            x_select = x0 + 1.1 * move_length * math.cos(angle_i * math.pi / 180)
            y_select = y0 + 1.1 * move_length * math.sin(angle_i * math.pi / 180)
            self.balls[i].expand(x_select, y_select)

    def shrink(self):
        self.is_expanded = False
        x0, y0 = self.pos().x() + (self.ball_size // 2), self.pos().y() + (
            self.ball_size // 2
        )
        for ball in self.balls:
            ball.shrink(x0, y0)

    def show(self) -> None:
        for ball in self.balls:
            ball.show()

        super(MainLandBall, self).show()

    def mouseMoveEvent(self, e):  # 重写移动事件
        x = e.x() - self.iniDragCor[0]
        y = e.y() - self.iniDragCor[1]
        cor = QPoint(x, y)

        if not (
            self.balls[0].anim is not None
            and str(self.balls[0].anim.state()) == "State.Running"
        ):
            self.move(self.mapToParent(cor))  # 需要maptoparent一下才可以的,否则只是相对位置。
            for i in self.balls:
                i.move_related(e)

    def mousePressEvent(self, e):
        self.move_start_time = time.time()
        if e.button() == Qt.LeftButton:
            self.iniDragCor[0] = e.x()
            self.iniDragCor[1] = e.y()
            for i in self.balls:
                i.update_position(e)

    def mouseReleaseEvent(self, event: PySide6.QtGui.QMouseEvent) -> None:
        if time.time() - self.move_start_time > 0.15:
            return
        self.expand(150) if not self.is_expanded else self.shrink()


if __name__ == "__main__":
    app = QApplication()
    win = MainLandBall(200)
    win.show()
    # win.expand(250)
    sys.exit(app.exec())
