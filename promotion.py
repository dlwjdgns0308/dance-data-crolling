import cv2
import mediapipe as mp
import time, os
import numpy as np
import math as mt

def getAngle(firstPoint,midPoint,lastPoint):
  result = np.degrees(mt.atan2(lastPoint[0][1] - midPoint[0][1],
  lastPoint[0][0]-midPoint[0][0])-mt.atan2(firstPoint[0][1]-midPoint[0][1],firstPoint[0][0]-midPoint[0][0]))
  result = abs(result)
  if (result>180):
    result = 360 - result
  return result
def getNeckAngle(ear,shoulder):

        result = np.degrees(mt.atan2(shoulder[0][1] - shoulder[0][1],
            (shoulder[0][0] + 100 )- shoulder[0][0])
                - mt.atan2(ear[0][1] - shoulder[0][1],
            ear[0][0] - shoulder[0][0]))

        result = abs(result) # 각도는 절대 음수일 수 없습니다

        if (result > 180) :
            result = 360.0 - result # 항상 각도를 선명하게 표현하십시오.
        return result
    

actions = ['dance']
seq_length = 30
secs_for_action = 60

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# For webcam input:

dance = "./dance.mp4"
cap = cv2.VideoCapture(dance)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
  for idx, action in enumerate(actions):
    data = []
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)
    cv2.waitKey(3000)
    start_time = time.time()

    while time.time() - start_time < secs_for_action:
      ret, img = cap.read()
      
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      result = pose.process(img)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      if result.pose_landmarks is not None:
        for res in [result.pose_landmarks]:
          joint = np.zeros((33,4))
          for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
          # Compute angles between joints
          v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
          v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
          v = v2 - v1 # [20, 3]
          #왼쪽 목
          leftNeckAngle = getNeckAngle(joint[[7],:3],joint[[11],:3])
          #오른쪽 목
          rightNeckAngle = getNeckAngle(joint[[7],:3],joint[[11],:3])
          # 왼쪽 가슴
          rightArmAngle = getAngle(joint[[11],:3],joint[[23],:3],joint[[25],:3])
          # 오른쪽 가슴
          rightChestAngle = getAngle(joint[[12],:3],joint[[24],:3],joint[[26],:3])
          # 왼쪽 다리
          rightArmAngle = getAngle(joint[[23],:3],joint[[25],:3],joint[[27],:3])
          # 오른쪽 다리
          rightArmAngle = getAngle(joint[[24],:3],joint[[26],:3],joint[[28],:3])
          # 왼쪽 팔
          rightArmAngle = getAngle(joint[[11],:3],joint[[13],:3],joint[[15],:3])
          # 오른쪽 팔
          rightArmAngle = getAngle(joint[[12],:3],joint[[14],:3],joint[[16],:3])
          

         # Get angle using arcos of dot product
          angle = np.arccos(np.einsum('nt,nt->n',
          v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
          v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

          angle = np.degrees(angle) # Convert radian to degree

          angle_label = np.array([angle], dtype=np.float32)
          angle_label = np.append(angle_label, idx)

          d = np.concatenate([joint.flatten(), angle_label])

          data.append(d)
      # Draw landmark annotation on the image.
          mp_drawing.draw_landmarks(
              img,
              result.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles
              .get_default_pose_landmarks_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('img', img)
      if cv2.waitKey(1) == ord('q'):
        break
    data = np.array(data)
    print(action, data.shape)
    np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

    # Create sequence data
    # full_seq_data = []
    # for seq in range(len(data) - seq_length):
    #   full_seq_data.append(data[seq:seq + seq_length])

    # full_seq_data = np.array(full_seq_data)
    # print(action, full_seq_data.shape)
    # np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
  break      