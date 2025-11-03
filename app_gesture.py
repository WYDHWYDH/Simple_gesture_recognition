import time
import random
from Infer import Inferhand
import cv2
import mediapipe as mp
import math
import numpy as np
import configparser
import data
mp_hands = mp.solutions.hands
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, \
    QScrollArea, QWidget, QSizePolicy, QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QSize, Slot, QTimer, QEventLoop, QThread
from PySide6.QtGui import QPixmap, QPainter, QFont

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_drawing = mp.solutions.drawing_utils


class GestureHandler:
    def __init__(self, image_viewer, text_display, main_window):
        self.image_viewer = image_viewer
        self.text_display = text_display
        self.main_window = main_window
        # self.gesture_active = False

    def handle_gesture(self, gesture):
        if gesture == "magnify":
            self.image_viewer.zoomIn()
        elif gesture == "shrink":
            self.image_viewer.zoomOut()
        elif gesture == "drag_up" or gesture == "drag_left":
            self.image_viewer.verticalScrollBar().setValue(self.image_viewer.verticalScrollBar().value() - 5)
        elif gesture == "drag_down" or gesture == "drag_right":
            self.image_viewer.verticalScrollBar().setValue(self.image_viewer.verticalScrollBar().value() + 5)
        elif gesture == "up_swipe" or gesture == "left_swipe":
            self.image_viewer.previousImage()
        elif gesture == "down_swipe" or gesture == "right_swipe":
            self.image_viewer.nextImage()

    def activate_gesture_control(self):
        self.gesture_active = True

    def deactivate_gesture_control(self):
        self.gesture_active = False


class TextDisplay(QTextEdit):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


class ImageViewer(QGraphicsView):
    def __init__(self, images):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scale_factor = 1.0
        self.zoom_factor = 1.0
        self.images = images
        self.current_image_index = 0
        self.loadImage()

    def loadImage(self):
        if 0 <= self.current_image_index < len(self.images):
            pixmap = QPixmap(self.images[self.current_image_index])
            item = QGraphicsPixmapItem(pixmap)
            self.scene.clear()
            self.scene.addItem(item)
            self.fitInView(item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

    @Slot()
    def zoomIn(self):
        self.scale(1.06)
        # self.scale(2)

    @Slot()
    def zoomOut(self):
        # self.scale(0.998)
        self.scale(0.925)

    @Slot(float)
    def scale(self, factor):
        self.zoom_factor *= factor
        self.scale_factor *= factor
        transform = self.transform()
        transform.scale(factor, factor)
        self.setTransform(transform)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(1.25, 1.25)
        else:
            self.scale(0.8, 0.8)

    def mousePressEvent(self, event):
        self.mouse_press_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self.mouse_release_pos = event.pos()
        if self.mouse_press_pos and self.mouse_release_pos:
            diff = self.mouse_release_pos - self.mouse_press_pos
            if diff.manhattanLength() > 3:
                if diff.x() > 0:
                    self.nextImage()
                else:
                    self.previousImage()

    @Slot()
    def nextImage(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.loadImage()

    @Slot()
    def previousImage(self):
        self.current_image_index = (self.current_image_index - 1 + len(self.images)) % len(self.images)
        self.loadImage()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(2,cv2.CAP_DSHOW)
        self.ups_openCount = 0
        self.state = False
        self.state_type = 0
        self.Action_correspondence = {"lr_open": ["left_swipe", "right_swipe"], "ups_open": "up_swipe",
                                      "dns_open": "down_swipe", "magnify": "magnify", "shrink": "shrink"}
        self.open_state = {"down": False, "left_right": False,
                           "zoom": False, "move": False, "up": False}
        self.open_count = {"down": 0, "left_right": 0,
                           "zoom": 0, "move": 0, "up": 0}
        self.open_wait_count = {"down": 0, "left_right": 0,
                                "zoom": 0, "move": 0, "up": 0}
        self.open_zh = {"down": "下挥准备", "left_right": "左右挥准备",
                        "zoom": "进入缩放模式", "move": "平移模式中", "up": "上挥准备"}
        self.ups_countine = 0
        self.ups_countine_temp = 0
        self.smcount = 0
        self.not_sm = 0
        self.frame_sequences = []
        self.closest_left_hand_index = -1  # 在主循环外初始化
        self.closest_left_hand_data = []  # 在主循环外初始化最近左手的数据列表
        self.frame_count = 0
        self.initUI()

        self.detect()

    @staticmethod
    def read_from_ini(file_path, section, key):
        config = configparser.ConfigParser()
        config.read(file_path, encoding='utf-8')
        if config.has_section(section) and config.has_option(section, key):
            return config.get(section, key)
        return ""
    def initUI(self):

        images = ['img01.jpg', 'img02.jpg', 'img03.jpg', 'img04.jpg']
        with open('test.txt', 'r', encoding='utf-8') as file:
            text_content = file.read()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 添加顶部布局和两个Label
        top_layout = QHBoxLayout()
        # 左标签
        self.mode_label = QLabel(" ")
        # 右标签
        self.mode_label.setFont(QFont("Arial", 24))
        self.status_label = QLabel("空闲")  # 使用self以方便后续更新
        self.status_label.setFont(QFont("Arial", 24))
        top_layout.addWidget(self.mode_label)
        top_layout.addWidget(self.status_label)
        main_layout.addLayout(top_layout)
        content_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        # 地球图片显示
        image_viewer = ImageViewer(images)
        scroll_area = QScrollArea()
        scroll_area.setWidget(image_viewer)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)
        prev_button = QPushButton('上一页')
        prev_button.clicked.connect(image_viewer.previousImage)
        button_layout.addWidget(prev_button)
        next_button = QPushButton('下一页')
        next_button.clicked.connect(image_viewer.nextImage)
        button_layout.addWidget(next_button)
        zoom_in_button = QPushButton('放大')
        zoom_in_button.clicked.connect(image_viewer.zoomIn)
        button_layout.addWidget(zoom_in_button)
        zoom_out_button = QPushButton('缩小')
        zoom_out_button.clicked.connect(image_viewer.zoomOut)
        button_layout.addWidget(zoom_out_button)
        scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content_layout.addLayout(left_layout)
        text_display = TextDisplay(text_content)
        text_display.setMinimumWidth(200)
        text_display.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        content_layout.addWidget(text_display)
        main_layout.addLayout(content_layout)
        self.gesture_handler = GestureHandler(image_viewer, text_display, self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.cleardownupleftright)
        self.wait_count = False
        self.setWindowTitle('手势操作界面')
        self.setGeometry(100, 100, 1000, 600)
        self.show()

    def process_frame(self, image):

        global closest_left_hand_index, closest_left_hand_data
        h, w, _ = image.shape
        black_frame = np.full((h, w, 3), 127,dtype=np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        closest_left_z = None
        closest_left_hand_data = None  # 初始化最近左手的数据列表
        closest_left_hand_index = -1

        if results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[index].classification[0].label
                if handedness == "Left":
                    MIDDLE_FINGER_MCP_Z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z * 1000
                    # 初始化或更新最近的左手MIDDLE_FINGER_MCP Z坐标
                    if closest_left_z is None or MIDDLE_FINGER_MCP_Z < closest_left_z:
                        closest_left_z = MIDDLE_FINGER_MCP_Z
                        closest_left_hand_index = index
                        closest_left_hand_data = hand_landmarks  # 清空旧数据，准备存储新数据

        # 绘制关键点
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[index].classification[0].label
                if handedness == "Left" and index == closest_left_hand_index:  # 绘制最近的左手
                    landmark_color = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

                    mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              landmark_drawing_spec=landmark_color,
                                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                             thickness=2,
                                                                                             circle_radius=0))

        return black_frame, closest_left_hand_data

    def clearLabels(self):
        self.status_label.setText(' ')
        self.mode_label.setText(' ')

    def cleardownupleftright(self):
        for item in ["down", "up", "left_right"]:
            self.open_count[item] = 0
            self.open_state[item] = False
        self.state = False
        self.wait_count = False
        self.status_label.setText(' ')
        self.mode_label.setText(' ')

        max_value = -1  #
        max_key = None
        for key, value in self.open_wait_count.items():
            if value > 15 and value > max_value:
                max_key = key

        if max_key is not None:
            self.open_state[max_key] = True
            self.state = True
            if max_key in ["down", "up", "left_right"]:
                self.timer.start(2000)
                self.wait_count = True
                if max_key == "left_right":
                    self.left_right_position = data.finger_x[2]
            elif max_key in ["zoom", "five", "fist","move"]:
                self.timer.stop()
            if max_key == "zoom" or max_key == "five" or max_key == "fist":
                self.anger_zoom_mean = sum(data.fingerOriginDiff_y[1:5]) / 4
                self.pre_originX, self.pre_originY = data.originX, data.originY
            self.status_label.setText(self.open_zh[max_key])

        for item in ["down", "up", "left_right", "zoom", "move"]:
            self.open_wait_count[item] = 0


    def detect(self):

        none_count = 0
        no_zoom_count = 0
        detect = Inferhand("example_model.onnx")
        while True:
            ret, frame = self.cap.read()
            cv2.imshow('video', frame)
            #print(frame.shape)
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame, hand_landmarks_ = self.process_frame(frame)
            if hand_landmarks_ != None:
                action = detect.detect(hand_landmarks_)
                print(action)
                data.UpdateData(hand_landmarks_, mp_hands, frame)
                if (action != None and not self.state):
                    if action == "fist" or action == "five":
                        self.open_count["zoom"] += 1
                        if self.open_count["zoom"] >8:
                            for item in ["down", "up", "left_right", "zoom", "move"]:
                                self.open_count[item] = 0
                            self.open_state["zoom"] = True
                            self.state = True
                    else:
                        self.open_count[action] += 1
                        if self.open_count[action] >8:
                            for item in ["down", "up", "left_right", "zoom", "move"]:
                                self.open_count[item] = 0
                            self.open_state[action] = True
                            self.state = True
                            if action in ["down", "up", "left_right"]:
                                self.timer.start(2000)
                                self.wait_count = True
                    if self.state:
                        if action == "fist" or action == "five":
                            action = "zoom"
                            print(self.open_zh["zoom"])
                        else:
                            print(self.open_zh[action])
                        self.status_label.setText(self.open_zh[action])
                        if action == "left_right":
                            self.left_right_position = data.finger_x[2]
                        elif action == "zoom" or action == "five" or action == "fist":
                            self.anger_zoom_mean = sum(data.fingerOriginDiff_y[1:5]) / 4
                            self.pre_originX,self.pre_originY = data.originX,data.originY

                if self.open_state["down"]:
                    if data.fingerAverage_y > hand_landmarks_.landmark[mp_hands.HandLandmark(9).value].y:
                        print("执行下挥")
                        self.gesture_handler.handle_gesture("down_swipe")
                        self.status_label.setText('执行下挥')
                        self.timer.stop()
                        QTimer.singleShot(300, self.clearLabels)
                        self.open_state["down"] = False
                        self.state = False
                        pre_action = None
                        self.open_count["up"] = -10
                        self.open_count["zoom"] = -10
                elif self.open_state["up"]:
                    if data.fingerAverage_y < hand_landmarks_.landmark[mp_hands.HandLandmark(9).value].y:
                        print("执行上挥")
                        self.status_label.setText('执行上挥')
                        self.gesture_handler.handle_gesture("up_swipe")
                        self.timer.stop()
                        QTimer.singleShot(300, self.clearLabels)
                        self.open_state["up"] = False
                        self.state = False
                        pre_action = None
                        self.open_count["down"] = -10
                        self.open_count["zoom"] = -10
                elif self.open_state["left_right"]:
                    #distence = [data.originX -data.finger_x[i] for i in range(1, 5)]
                    New_position = data.finger_x[2]

                    drift = self.left_right_position - abs(New_position)
                    if (abs(drift) > 0.15):
                        print("执行左挥" if drift > 0 else "执行右挥")
                        self.gesture_handler.handle_gesture("left_swipe" if drift > 0 else "right_swipe")
                        self.status_label.setText("执行左挥" if drift > 0 else "执行右挥")
                        self.timer.stop()
                        QTimer.singleShot(300, self.clearLabels)
                        for item in ["down", "up", "left_right", "zoom", "move"]:
                            self.open_count[item] = 0
                            self.open_state[item] = False
                        self.state = False
                        pre_action = None
                elif self.open_state["zoom"]:
                    #print(abs(self.pre_originX-data.originX))
                    if action in ["five","fist","zoom"] and abs(self.pre_originX-data.originX)<0.1 and abs(self.pre_originY-data.originY)<0.1:
                        anger = sum(data.fingerOriginDiff_y[1:5]) / 4
                        # print(anger, anger_zoom_mean)
                        if action == "five":
                            self.gesture_handler.handle_gesture("magnify")
                            self.status_label.setText('持续放大')
                        elif action == "fist":
                            self.gesture_handler.handle_gesture("shrink")
                            self.status_label.setText('持续缩小')
                        elif abs(anger - self.anger_zoom_mean) >= 0.003:
                            if abs(anger) > abs(self.anger_zoom_mean):
                                self.gesture_handler.handle_gesture("magnify")
                                self.status_label.setText('缩放中')
                            else:
                                self.gesture_handler.handle_gesture("shrink")
                                self.status_label.setText('缩放中')
                            self.anger_zoom_mean = anger
                    else:
                        no_zoom_count = no_zoom_count + 1
                        if no_zoom_count == 10:
                            self.status_label.setText('停止缩放')
                            QTimer.singleShot(300, self.clearLabels)
                            for item in ["down", "up", "left_right", "zoom", "move"]:
                                self.open_count[item] = 0
                                self.open_state[item] = False
                            self.state = False
                            no_zoom_count = 0
                elif self.open_state["move"]:
                    if action == "move":
                        self.status_label.setText("正在平移")
                        v_scroll = self.gesture_handler.image_viewer.verticalScrollBar()
                        h_scroll = self.gesture_handler.image_viewer.horizontalScrollBar()
                        real_v = hand_landmarks_.landmark[mp_hands.HandLandmark(16).value].x * 640 - 80
                        real_v = real_v / 480 if real_v > 0 else 0
                        real_h = hand_landmarks_.landmark[mp_hands.HandLandmark(16).value].y * 480 - 100
                        real_h = real_h / 280 if real_h > 0 else 0
                        h_scroll.setValue(
                            real_v * h_scroll.maximum())
                        v_scroll.setValue(
                            real_h * v_scroll.maximum())
                    else:
                        for item in ["down", "up", "left_right", "zoom", "move"]:
                            self.open_count[item] = 0
                            self.open_state[item] = False
                        self.state = False
                        self.status_label.setText("停止平移")
                        QTimer.singleShot(300, self.clearLabels)
                if self.wait_count and action != None:
                    if action == "fist" or action == "five":
                        self.open_wait_count["zoom"] += 1
                    else:
                        self.open_wait_count[action] += 1
            else:
                none_count = none_count +1
                if none_count==10:
                    for item in ["down", "up", "left_right", "zoom", "move"]:
                        self.open_count[item] = 0
                        self.open_wait_count[item] = 0
                        self.open_state[item] = False
                    self.state = False
                    self.wait_count = False
                    print("已经清空")
                    self.timer.stop()
                    self.status_label.setText(' ')
                    self.mode_label.setText(' ')
                    none_count = 0
            cv2.imshow('MediaPipe Hands', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                self.cap.__del__()
                break


    def closeEvent(self,event):
        self.cap.release()
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
