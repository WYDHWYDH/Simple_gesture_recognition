import numpy as np
import onnxruntime as ort
import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)


class AisInferhand():
    IndexPredicted = {0: None, 1: "down", 2: "up", 3: "zoom", 4: "left_right", 5: "five", 6: "fist", 7: "move"}

    def __init__(self, om_path):
        self.model = InferSession(0, om_path)
        # self.input_name = self.session.get_inputs()[0].name
        # self.output_name = self.session.get_outputs()[0].name

    def detect(self, Hand_key_points):
        if Hand_key_points == None:
            return None
        Hand_key_points_yuan = Hand_key_points
        Hand_key_points = self.landmarks_to_numpy(Hand_key_points)
        Hand_key_points = self.process_mark_data(Hand_key_points)
        Hand_key_points = [Hand_key_points]
        Hand_key_points = np.array(Hand_key_points).astype(np.float32)
        #predicted = self.session.run([self.output_name], {self.input_name: Hand_key_points})
        predicted = self.model.infer(Hand_key_points)
        predicted = predicted[0].tolist()
        predicted = predicted[0].index(max(predicted[0]))
        if (AisInferhand.IndexPredicted[predicted]!="zoom"):
            return AisInferhand.IndexPredicted[predicted]
        gesture_fist = [
            all([Hand_key_points_yuan.landmark[mp_hands.HandLandmark(0).value].y > Hand_key_points_yuan.landmark[
                mp_hands.HandLandmark(i).value].y > Hand_key_points_yuan.landmark[mp_hands.HandLandmark(i - 3).value].y - 0.02
                 for i in [8, 12, 16, 20]]),
            all([Hand_key_points_yuan.landmark[mp_hands.HandLandmark(0).value].y > Hand_key_points_yuan.landmark[
                mp_hands.HandLandmark(i).value].y > Hand_key_points_yuan.landmark[mp_hands.HandLandmark(i - 2).value].y - 0.02
                 for i in [7, 11, 15, 19]]),
        ]
        if not False in gesture_fist:
            return "fist"
        else:
            return "zoom"
    @staticmethod
    def landmarks_to_numpy(results):
        return np.array(
            np.array([[results.landmark[i].x, results.landmark[i].y, results.landmark[i].z] for i in range(21)]))

    @classmethod
    def process_mark_data(cls, hand_arr):
        lh_root = hand_arr[0, 0]
        lh_marks = cls.relative_coordinate(hand_arr, lh_root)
        if lh_marks.all() != 0:
            lh_marks = cls.standardization(lh_marks)
        return np.array(lh_marks)

    @staticmethod
    def relative_coordinate(arr, point):
        return arr - point

    @staticmethod
    def standardization(hand_arr):
        return (hand_arr - np.mean(hand_arr)) / np.std(hand_arr)


def process_frame(image):
    global closest_left_hand_index, closest_left_hand_data
    h, w, _ = image.shape
    black_frame = np.zeros((h, w, 3), dtype=np.int8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pt=time.time()
    results = hands.process(image_rgb)
    print(time.time()-pt)
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


if __name__ == '__main__':
    cap = cv2.VideoCapture(4)
    detect = AisInferhand("model.om")
    pretime=0
    while True:
        flag, frame = cap.read()
        if flag:
            frame = cv2.flip(frame, 1)
            frame, result = process_frame(frame)
            cv2.imshow("frame", frame)
            action = detect.detect(result)
            print(action)
            print(time.time()-pretime)
            pretime = time.time()
            key =  cv2.waitKey(1)
            if key==ord("q"):
                break

