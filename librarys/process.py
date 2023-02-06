import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from pynput.mouse import Button
from PySide6.QtCore import QObject, Signal

sys.path.append("./librarys")

from widgets import *  # noqa

from . import config_values as cfg  # noqa

global widgets


class GRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layer, num_classes):
        super(GRU, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer  # GRU网络层数
        self.lstm = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            batch_first=True,
            dropout=0.5,
        )
        self.relu = nn.PReLU()  # PReLU激活函数，防止死亡ReLU问题
        self.classes = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=num_classes
            ),  # num_classes为分类数
        )

    def forward(self, x):
        x, h_0 = x  # 将输入拆分
        batch_size = x.shape[0]  # 获取x的batch_size
        out, h_t1 = self.lstm(x, h_0)  # 将数据传入GRU网络训练
        out = h_t1[-1:, :, :]  # 取得最后一层GRU的输出
        out = out.view(batch_size, -1)  # 将维度从(1, b, hiddem) => (b, hiddem)
        out = self.classes(out)  # 进入全连接层训练
        out = self.relu(out)  # 激活输出
        return out, h_t1  # 返回out输出及h_t给下一层


class Identify(QObject):
    control_ball_open_and_close = Signal(bool)

    def __init__(self, win):
        super().__init__()
        self.prin_time = None
        self.h_t = None
        self.win = win
        self.isEnd = False
        self.in_dim_stack = deque(maxlen=30)

    def start(self):
        threading.Thread(target=self.run).start()

    def run(self):
        self.cap = cv2.VideoCapture(0)  # 摄像头图像采集
        movement = {
            0: "点击",
            1: "平移",
            2: "缩放",
            3: "抓取",
            4: "旋转",
            5: "无",
            6: "截图",
            7: "放大",
        }
        S = 0  # 每帧的处理时间
        device = torch.device("cpu")  # 初始化于cpu上处理
        if torch.cuda.is_available():  # 判断是否能使用cuda
            device = torch.device("cuda:0")
        model = torch.load(r"model_cnn.pt", map_location="cpu").to(device)  # 载入模型
        # print(model)
        hiddem_dim = 30  # 隐藏层大小
        num_layers = 2  # GRU层数

        self.h_t = torch.zeros(num_layers, 1, hiddem_dim)  # 初始化全0的h_t
        self.h_t = self.h_t.to(device)  # 载入设备

        last_gesture = "无"  # 初始化最后输出，用于判断是否与当前输出一致
        self.prin_time = time.time()  # 初始化输出时间

        mp_drawing = cfg.detector.mpDraw
        mp_hands = cfg.detector.mpHands
        ratio = self.cap.get(4) / self.cap.get(3)  # 高宽比

        start_time = time.time()  # 初始化当前帧帧起始时间

        opened = self.cap.isOpened()

        selecting = False

        while opened:
            if self.isEnd:
                break
            self.win.eventRunning.wait()

            in_dim = torch.zeros(126)  # 使得一开始的帧为全0

            # 判断是否满足当前帧率
            wait_time = S - (time.time() - start_time)
            if wait_time > 0:
                time.sleep(wait_time)
            start_time = time.time()  # 重置起始时间

            success, image = self.cap.read()  # 获取摄像头输出
            size = (
                int(self.win.ui.label_img.width()),
                int(self.win.ui.label_img.width() * ratio),
            )
            cfg.wCam = size[0]
            cfg.hCam = size[1]
            image = cv2.resize(image, size)

            image = cv2.flip(image, 1)  # 水平翻转
            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            if not success:
                break

            all_hands, _ = cfg.detector.findHands(image)

            lmList, bbox, depth_radius = cfg.detector.findPosition(_)

            _, in_dim = self.draw_finger_and_get_indim(
                image, in_dim, mp_drawing, mp_hands, cfg.detector.hands
            )
            self.in_dim_stack.append(in_dim)    

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.reaction(
                device,
                self.h_t,
                hiddem_dim,
                in_dim,
                last_gesture,
                model,
                movement,
                num_layers,
                self.prin_time,
            )
            # 取食指和中指的尖端
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                # 3. Check which fingers are up
                fingers = cfg.detector.fingersUp(lmList)
                print(fingers)
                cv2.rectangle(
                    image,
                    (cfg.frameR, cfg.frameR),
                    (cfg.wCam - cfg.frameR, cfg.hCam - cfg.frameR),
                    (255, 0, 255),
                    2,
                )

                if cfg.NOW_MODE == cfg.MOVE_AND_CLICK:
                    image = self.MOVE_AND_CLICK_FUNC(
                        fingers, cfg.hCam, image, cfg.wCam, x1, y1
                    )

                elif cfg.NOW_MODE == cfg.PPT_WRITE:
                    self.PPT_WRITER_FUNC(cfg.hCam, image, lmList, cfg.wCam)

                # Selecting Mode
                if fingers == [0, 1, 1, 1, 1]:
                    if cfg.bef_selecting == 0:
                        cfg.bef_selecting = time.time()
                    if time.time() - cfg.bef_selecting > 0.8:
                        cfg.NOW_MODE, cfg.NOW_MODE_COLOR = self.select_mode(
                            image, lmList, cfg.offset, x2, y2
                        )
                        selecting = True
                else:
                    cfg.bef_selecting = 0
                    selecting = False

                print(selecting, self.win.main_land_ball.is_expanded)

            if self.win.eventRunning.isSet():
                self.win.flash_img(image, ratio)
                if selecting != self.win.main_land_ball.is_expanded:
                    self.control_ball_open_and_close.emit(selecting)

        self.cap.release()

    def PPT_WRITER_FUNC(self, hCam, img, lmList, wCam):
        # 食指和大拇指的坐标
        x_shi, y_shi = lmList[8][1:]
        x_damu, y_damu = lmList[4][1:]
        # 两个手指之间连线
        cx = (x_shi + x_damu) // 2
        cy = (y_shi + y_damu) // 2
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        # 两个手指之间的距离
        length = math.hypot(x_damu - x_shi, y_damu - y_shi)
        print(length)
        # 中点
        cv2.rectangle(
            img,
            (cfg.frameR, cfg.frameR),
            (wCam - cfg.frameR, hCam - cfg.frameR),
            (255, 0, 255),
            2,
        )
        x3 = np.interp(cx, (cfg.frameR, wCam - cfg.frameR), (0, cfg.wScr))
        y3 = np.interp(cy, (cfg.frameR, hCam - cfg.frameR), (0, cfg.hScr))
        # x3 = np.interp(x1, (wCam - cfg.frameR, cfg.frameR), (0, wScr))
        # y3 = np.interp(y1, (hCam - cfg.frameR, cfg.frameR), (0, hScr))
        # Smoothen Values
        clocX = cfg.plocX + (x3 - cfg.plocX) / cfg.smoothening
        clocY = cfg.plocY + (y3 - cfg.plocY) / cfg.smoothening
        # mouse move
        cfg.mouse_points.append((cfg.wScr - x3, y3))
        xx3_sum, yy3_sum = 0, 0

        for xx3, yy3 in cfg.mouse_points:
            xx3_sum += xx3
            yy3_sum += yy3

        cfg.mouse.position = (xx3_sum // cfg.smoothening, yy3_sum // cfg.smoothening)
        cfg.plocX, cfg.plocY = clocX, clocY
        if length <= 25:
            # 模拟左键一直按下
            # 如果左键已经按下，就不再按下
            if not cfg.leftDown:
                cfg.mouse.press(Button.left)
                cfg.leftDown = True
        if length > 45:
            # 模拟左键松开
            if cfg.leftDown:
                cfg.mouse.release(Button.left)
                cfg.leftDown = False

    def MOVE_AND_CLICK_FUNC(self, fingers, hCam, img, wCam, x1, y1):
        # 一根手指：移动指针
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.rectangle(
                img,
                (cfg.frameR, cfg.frameR),
                (wCam - cfg.frameR, hCam - cfg.frameR),
                (255, 0, 255),
                2,
            )
            x3 = np.interp(x1, (cfg.frameR, wCam - cfg.frameR), (0, cfg.wScr))
            y3 = np.interp(y1, (cfg.frameR, hCam - cfg.frameR), (0, cfg.hScr))
            # x3 = np.interp(x1, (wCam - cfg.frameR, cfg.frameR), (0, wScr))
            # y3 = np.interp(y1, (hCam - cfg.frameR, cfg.frameR), (0, hScr))

            # Smoothen Values
            clocX = cfg.plocX + (x3 - cfg.plocX) / cfg.smoothening
            clocY = cfg.plocY + (y3 - cfg.plocY) / cfg.smoothening
            # mouse move

            cfg.mouse_points.append((x3, y3))
            xx3_sum, yy3_sum = 0, 0
            for xx3, yy3 in cfg.mouse_points:
                xx3_sum += xx3
                yy3_sum += yy3
            cfg.mouse.position = (
                cfg.wScr - (xx3_sum // cfg.smoothening),
                yy3_sum // cfg.smoothening,
            )
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cfg.plocX, cfg.plocY = clocX, clocY
        if fingers[1] == 1 and fingers[2] == 1:
            leng, img, _ = cfg.detector.findDistance(8, 12, img)
            print(leng)
            if leng < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                if time.time() - cfg.bef_clicked > 0.5:
                    cfg.mouse.click(Button.left, 1)
                    cfg.bef_clicked = time.time()
        # if fingers == [1, 1, 1, 1, 1]:
        #     # 模拟Enter
        #     if time.time() - bef_clicked > 0.5:
        #         keyboard.press(pynput.keyboard.Key.enter)
        #         bef_clicked = time.time()
        return img

    def select_mode(self, img, lmList, offset, x2, y2):
        # 获取手掌的位置 WRIST
        x0, y0 = lmList[0][1:]
        # 在 x0, y0 建立极坐标系
        # 计算x0, y0 到 x2, y2 的角度
        angle = math.atan2(y2 - y0, x2 - x0) * 180 / math.pi
        # 计算x0, y0 到 x2, y2 的距离
        distance = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
        print(f"Angle = {angle}")
        # 求解x_mode, y_mode，其中，在x0, y0 建立的极坐标系中，x3, y3 与 x2, y2 的距离0.1 distance
        x_mode = x0 + 1.1 * distance * math.cos(angle * math.pi / 180)
        y_mode = y0 + 1.1 * distance * math.sin(angle * math.pi / 180)
        # 绘制模式切换的圆圈
        cv2.circle(img, (int(x_mode), int(y_mode)), 15, (0, 255, 0), cv2.FILLED)
        # 模式切换
        # 每隔15度就绘制1个圆圈，共绘制5个
        nearest = 10000000
        nearest_i_angle = -1
        nearest_i_color_deep = -1
        nearest_i_color = -1
        nearest_i_MODE = -1

        for i, (angle_i, color, color_deep) in enumerate(
            zip(
                range(0 + offset, int(12 * 5 + offset), 10),
                cfg.color_set,
                cfg.color_set_deep,
            )
        ):
            x_select = x0 + 1.1 * distance * math.cos(angle_i * math.pi / 180)
            y_select = y0 + 1.1 * distance * math.sin(angle_i * math.pi / 180)
            # 圆圈，不填充

            cv2.circle(
                img,
                (int(x_select), int(y_select)),
                25,
                (color[2], color[1], color[0]),
                thickness=10,
            )
            bef_near = nearest
            nearest = min(
                nearest, math.sqrt((x_mode - x_select) ** 2 + (y_mode - y_select) ** 2)
            )
            if nearest != bef_near:
                nearest_i_angle = angle_i
                nearest_i_color_deep = color_deep
                nearest_i_MODE = i
                nearest_i_color = color

        x_selected = x0 + 1.1 * distance * math.cos(nearest_i_angle * math.pi / 180)
        y_selected = y0 + 1.1 * distance * math.sin(nearest_i_angle * math.pi / 180)

        cv2.circle(
            img,
            (int(x_selected), int(y_selected)),
            25,
            (nearest_i_color_deep[2], nearest_i_color_deep[1], nearest_i_color_deep[0]),
            thickness=10,
        )
        return nearest_i_MODE, (
            nearest_i_color[2],
            nearest_i_color[1],
            nearest_i_color[0],
        )

    def reaction(
        self,
        device,
        h_t,
        hiddem_dim,
        in_dim,
        last_gesture,
        model,
        movement,
        num_layers,
        prin_time,
    ):
        if len(self.in_dim_stack) == 30:
            in_dims = list(self.in_dim_stack)
            in_dims = np.stack(in_dims, axis=0)
            in_dims = torch.from_numpy(in_dims).float()

            # in_dim = in_dim.unsqueeze(dim=0)
            in_dims = in_dims.unsqueeze(dim=0)
            print(in_dims.shape)
            in_dims = in_dims.to(torch.float32).to(device)
            self.h_t = self.h_t.to(torch.float32).to(device)
            if time.time() - prin_time < 2:
                in_dims = torch.zeros(1, 30, 126).to(device)
            # rel, self.h_t = model((in_dim, self.h_t))
            t1 = time.time()
            rel = model(in_dims)
            rel = torch.sigmoid(rel)
            print(time.time() - t1)
            # print(rel)
            confidence, rel = rel.max(1)
            # 对每个动作设置单独的置信度阈值
            cfd = {
                "点击": 0.48,
                "平移": 0.997,
                "缩放": 0.90,
                "抓取": 0.992,
                "旋转": 0.98,
                "无": 0,
                "截图": 0.87,
                "放大": 0.4,
            }
            print(movement[rel.item()], " \t置信度：", round(confidence.item(), 2))
            if confidence > cfd[movement[rel.item()]]:  # 超过阈值的动作将会被输出
                now_gesture = last_gesture
                last_gesture = movement[rel.item()]
                if not (now_gesture == last_gesture):  # 判断是否与上次的输出相同，若相同则不输出
                    if time.time() - prin_time > 2:  # 若距离上次输出时间小于2秒，则不输出

                        self.win.set_gesture(movement[rel.item()])
                        self.prin_time = time.time()  # 重置输出时间
                        self.h_t = torch.zeros(num_layers, 1, hiddem_dim).to(
                            device
                        )  # 将当前的h_t重置

    def draw_finger_and_get_indim(self, image, in_dim, mp_drawing, mp_hands, hands):
        image.flags.writeable = False
        results = cfg.detector.results
        # 将指关节点绘于image
        image.flags.writeable = True
        # 获取当前帧指关节点数据
        if results.multi_hand_landmarks:

            # 判断手掌个数
            if (
                len(results.multi_handedness) == 1
                and results.multi_handedness[0].classification.__getitem__(0).index == 1
            ):
                # print("Single Hand!!!!!!!!!!!!!!!")
                # 判断左右手先后
                for hand_landmarks in results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_1 = []
                    # 将指关节点数据依次存入index
                    for k in range(0, 21):
                        index_1.append(hand_landmarks.landmark[k].x)
                        index_1.append(hand_landmarks.landmark[k].y)
                        index_1.append(hand_landmarks.landmark[k].z)
                    for k_1 in range(0, 63):
                        index_1.append(0)
                # 最后将index（126）添加至in_dim（x，126）末尾
                in_dim = torch.from_numpy(np.array(index_1))
            elif (
                len(results.multi_handedness) == 1
                and results.multi_handedness[0].classification.__getitem__(0).index == 0
            ):
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    index_0 = []
                    for k_1 in range(0, 63):
                        index_0.append(0)
                    for k in range(0, 21):
                        index_0.append(hand_landmarks.landmark[k].x)
                        index_0.append(hand_landmarks.landmark[k].y)
                        index_0.append(hand_landmarks.landmark[k].z)
                in_dim = torch.from_numpy(np.array(index_0))
            elif (
                len(results.multi_handedness) == 2
                and results.multi_handedness[0].classification.__getitem__(0).index == 1
            ):
                index_1_first = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    for k in range(0, 21):
                        index_1_first.append(hand_landmarks.landmark[k].x)
                        index_1_first.append(hand_landmarks.landmark[k].y)
                        index_1_first.append(hand_landmarks.landmark[k].z)
                in_dim = torch.from_numpy(np.array(index_1_first))
            elif (
                len(results.multi_handedness) == 2
                and results.multi_handedness[0].classification.__getitem__(0).index == 0
            ):
                results.multi_hand_landmarks.reverse()
                index_0_first = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    for k in range(0, 21):
                        index_0_first.append(hand_landmarks.landmark[k].x)
                        index_0_first.append(hand_landmarks.landmark[k].y)
                        index_0_first.append(hand_landmarks.landmark[k].z)
                in_dim = torch.from_numpy(np.array(index_0_first))
        return image, in_dim

    def break_loop(self):
        self.isEnd = True
