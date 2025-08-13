#!/usr/bin/env python3
import cv2
import math
import time
from dynamixel_sdk import *
from ultralytics import YOLO

# -------------------- Dynamixel Setup -------------------- #
DEVICENAME = 'COM3'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
DXL_IDS = [1, 2, 3, 4]
TORQUE_ENABLE = 1

def radians_to_position(rad, scale=1.0):
    center_position = 512
    max_deviation = 511
    scaled_rad = (rad / math.pi) * scale
    goal_position = center_position + (scaled_rad * max_deviation)
    return max(0, min(1023, int(goal_position)))

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("❌ 포트 열기 실패"); exit()
if not portHandler.setBaudRate(BAUDRATE):
    print("❌ 보율 설정 실패"); exit()
print("✅ 포트 연결 완료")

DEFAULT_SPEED = 50
for dxl_id in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_MOVING_SPEED, DEFAULT_SPEED)
print(f"✅ 모든 모터의 기본 속도가 {DEFAULT_SPEED}로 설정되었습니다.")

# -------------------- YOLOv11 Pose 모델 로드 -------------------- #
model = YOLO("yolo11n-pose.pt")
model.to("cpu")

# -------------------- Camera 설정 -------------------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패"); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera_width, camera_height = 640, 480

# -------------------- 상태 변수 -------------------- #
joint_pos = [0.0, 0.0, 0.0, 0.0]
initial_face_height = None

def update_motors(pos_rad):
    for i, rad in enumerate(pos_rad):
        dxl_pos = radians_to_position(rad)
        packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_GOAL_POSITION, dxl_pos)

# -------------------- Main Loop -------------------- #
try:
    while True:
        start_time = time.time()  # ⏱️ 프레임 시작 시간

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=image, conf=0.5, iou=0.5, verbose=False)
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints.xy) > 0:
            kp = keypoints.xy[0]  # 첫 번째 사람
            kp = kp.cpu().numpy()

            # 마크 표시 (시각화)
            for i, (x, y) in enumerate(kp):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    cv2.putText(frame, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 주요 keypoint
            nose = kp[0]
            l_eye = kp[1]
            r_eye = kp[2]
            l_ear = kp[3]
            r_ear = kp[4]
            chin = kp[8]

            # 1. Joint 1: 얼굴 좌우 회전 추정 (Yaw)
            left_dist = abs(nose[0] - l_eye[0]) if l_eye[0] > 0 else None
            right_dist = abs(r_eye[0] - nose[0]) if r_eye[0] > 0 else None

            yaw_ratio = 0
            if left_dist and right_dist:
                yaw_ratio = (right_dist - left_dist) / (left_dist + right_dist + 1e-5)  # -1 ~ 1 범위
            elif l_ear[0] > 0 and r_ear[0] > 0:
                yaw_ratio = (r_ear[0] - l_ear[0]) / camera_width

            joint_pos[0] = max(-math.pi, min(math.pi, joint_pos[0] + yaw_ratio * 0.5))

            # 2. Joint 2: 거리 추정 (코와 턱 사이 y축 거리)
            current_height = abs(nose[1] - chin[1])
            if initial_face_height is None:
                initial_face_height = current_height
            dz = (initial_face_height - current_height) * 0.01

            # 3. Joint 3: 상하 위치 (코의 y좌표 기준 화면 중앙과의 차이)
            face_center_y = nose[1]
            center_y = camera_height // 2
            dy = (face_center_y - center_y) * 0.005

            # 4. Joint 4: Gaze 추정 (코 ↔ 왼쪽 눈 거리)
            if l_eye[0] > 0:
                gaze_direction = (nose[0] - l_eye[0]) / camera_width * 5.0
            else:
                gaze_direction = 0.0

            # 위치 적용
            joint_pos[1] = max(-1.5, min(1.5, joint_pos[1] + dz))
            joint_pos[2] = max(-1.5, min(1.5, joint_pos[2] + dy))
            joint_pos[3] = max(-1.0, min(1.0, joint_pos[3] + gaze_direction))

            update_motors(joint_pos)
            cv2.putText(frame, "Pose Tracked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            initial_face_height = None
            cv2.putText(frame, "No Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLOv11 Pose + Dynamixel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("🛑 종료 및 포트 닫힘")
    cap.release()
    cv2.destroyAllWindows()
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
