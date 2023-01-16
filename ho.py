import socket
import sys
import numpy as np
import mediapipe as mp
import cv2
import json

# import vec


def init_TCP():
    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    # address = ('172.16.61.87', port)
    # address = ('192.168.0.107', port)
    address = ('127.0.0.1', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.hostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(
            socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # print(socket.gethostbyname(socket.hostname()))
    # s.connect(address)
    # return s


def send_info_to_unity(s, args: str):
    # msg = '%.4f ' * len(args) % args

    try:
        # print(len(args))
        s.send(args.encode())
        # s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()


# global variable
port = 5066  # have to be same as unity

# 소켓 열기
# useSocket = init_TCP()

# 동영상 PATH
VIDEO_PATH = "JustDoIt.mp4"

# 동영상 불러오기
cap = cv2.VideoCapture(VIDEO_PATH)
# cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)
# cap.set(3, 1920)
# cap.set(4, 1080)

# 포즈에 저장 되어 있는 모든 Enums 데이터.
pose_and_face = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
# 왼손에 저장 되어 있는 모든 Enums 데이터
left_hand = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
             'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP',
             'RING_FINGER_TIP',
             'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
# 오른손에 저장 되어 있는 모든 Enums 데이터
right_hand = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2',
              'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
              'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2',
              'RING_FINGER_TIP2',
              'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']


def get_video_frame():
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    # img_h, img_w, _ = frame.shape 

    # return frame, img_h, img_w
    return frame


# 미디어파이 holistic 솔루션 불러오기 및 미디어파이에서 제공하는 drawing 불러와서 스켈레톤 그리기
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def run_holistic(frame):
    # 필요한 설정 초기화
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.7,
        model_complexity=1,
        enable_segmentation=False,
        min_tracking_confidence=0.7
    )

    # cv2에서는 RGB 쓰므로 BGR -> RGB 변환

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # holistic process 변환한 이미지를 넣고 시작
    results = holistic.process(img)

    # msg = []
    # if results.left_hand_landmarks:
    #     for i in range(len(left_hand)):
    #         msg.append(
    #             results.left_hand_landmarks.landmark[i].x,
    #             results.left_hand_landmarks.landmark[i].y,
    #             results.left_hand_landmarks.landmark[i].z
    #         )

    # print(msg)

    # data = {"pose": []}
    lm = {}
    print(dir(results))
    if results.pose_landmarks:
        for i in range(len(pose_and_face)):
            # lm.update({
            #     pose_and_face[i]: {
            #         "x": round(results.pose_world_landmarks.landmark[i].x, 4),
            #         "y": round(results.pose_world_landmarks.landmark[i].y, 4),
            #         "z": round(results.pose_world_landmarks.landmark[i].z, 4)
            #     }})
            lm.update({
                pose_and_face[i]: {
                    "x": results.pose_world_landmarks.landmark[i].x,
                    "y": results.pose_world_landmarks.landmark[i].y,
                    "z": results.pose_world_landmarks.landmark[i].z
                }})
            # lm.update({
            #     pose_and_face[i] :{
            #     "x": results.pose_landmarks.landmark[i].x * frame[2],
            #     "y": results.pose_landmarks.landmark[i].y * frame[1],
            #     "z": results.pose_landmarks.landmark[i].z 
            # }})
            # lm.append({
            #     pose_and_face[i] :{
            #     "x": round(results.pose_landmarks.landmark[i].x, 4),
            #     "y": round(results.pose_landmarks.landmark[i].y, 4),
            #     "z": round(results.pose_landmarks.landmark[i].z, 4)
            # }})

    print(lm)
    # if len(lm) != 0:
    #     pose = vec.vecData(lm)

    #     print(pose['UpperArm']['r'])
    # if len(lm) != 0:
    #     msg = json.dumps(lm)
    #     send_info_to_unity(useSocket, msg)
    # data.update({"pose": lm})

    # print(data)
    # msg = {"pose":[]}

    annotated_image = frame.copy()

    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.left_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS)
    #
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.right_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS)

    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.face_landmarks,
    #     mp_holistic.FACE MESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_tesselation_style())

    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())

    cv2.imshow("image", annotated_image)


while True:

    frame = get_video_frame()

    run_holistic(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
