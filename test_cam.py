#!/usr/bin/env python3
# 필요한 라이브러리들을 가져옵니다.
import cv2
import math
import time
from dynamixel_sdk import *
import mediapipe as mp
import numpy as np

# -------------------- Dynamixel Setup (다이나믹셀 모터 설정) -------------------- #
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32

DXL_IDS = [1, 2, 3, 4]
TORQUE_ENABLE = 1

# --- 조인트별 초기 위치 및 가동범위 설정 ---
INITIAL_POSITIONS = [514, 300, 600, 600]
POSITION_LIMITS = [
    [170, 850],  # joint1 (R-limit, L-limit)
    [60, 520],   # joint2 (R-limit, L-limit)
    [260, 780],  # joint3 (R-limit, L-limit) 
    [200, 820],  # joint4 (R-limit, L-limit)
]
# ---------------------------------------------------------

# --- 제어 민감도 및 임계값 설정 ---
ROTATION_SPEED = 0.03  # Joint1 회전 속도 (단위: 라디안/프레임) - 값을 약간 올렸습니다.
VERTICAL_SPEED = 0.02 # J2, J3의 상하 이동 속도
# -----------------------------------------------

def dxl_to_radians(dxl_pos):
    return (dxl_pos - 512) * (math.pi / 512)

def radians_to_position(rad):
    center_position = 512
    max_deviation = 511
    goal_position = center_position + (rad / math.pi) * max_deviation
    return max(0, min(1023, int(goal_position)))

# -------------------- Port/Packet Handler Initialization -------------------- #
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort(): print("❌ 포트 열기 실패"); exit()
if not portHandler.setBaudRate(BAUDRATE): print("❌ 보율 설정 실패"); exit()
print("✅ 포트 연결 완료")

# --- 초기화 ---
DEFAULT_SPEED = 50
for dxl_id in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_MOVING_SPEED, DEFAULT_SPEED)
print(f"✅ 모든 모터의 기본 속도가 {DEFAULT_SPEED}로 설정되었습니다.")

print("ℹ️ 초기 자세로 이동합니다...")
for i, dxl_id in enumerate(DXL_IDS):
    packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, INITIAL_POSITIONS[i])

initial_joint_pos_rad = [dxl_to_radians(pos) for pos in INITIAL_POSITIONS]

print(f"ℹ️ 5초간 초기 자세를 유지합니다...")
time.sleep(5)
print("✅ 얼굴 추적을 시작합니다.")

# -------------------- Face Tracker 설정 -------------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0) # 실시간 웹캠 사용
if not cap.isOpened(): print("❌ 카메라 열기 실패"); exit()

camera_width, camera_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

# -------------------- Joint Position 변수 설정 -------------------- #
joint_pos = list(initial_joint_pos_rad)

def update_motors(pos_rad):
    for i, rad in enumerate(pos_rad):
        dxl_pos = radians_to_position(rad)
        limit_r, limit_l = POSITION_LIMITS[i]
        final_pos = max(limit_r, min(limit_l, dxl_pos))
        packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_GOAL_POSITION, final_pos)

# -------------------- Main Loop -------------------- #
try:
    while True:
        ret, frame = cap.read()
        if not ret: print("카메라 입력 실패"); break
        frame = cv2.flip(frame, 1) # 좌우 반전 (거울 모드)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        orientation_text = "Forward" # 기본 텍스트

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- 조인트별 추적 계산 (요청사항 반영된 최종 로직) ---

            # 1. Joint 1: Yaw 추정 및 회전 (새로운 방식)
            nose_tip = landmarks[1]
            left_eye_inner = landmarks[226]
            right_eye_inner = landmarks[446]

            eye_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
            yaw_deviation = (nose_tip.x - eye_center_x) * camera_width
            yaw_threshold = 20  # 정면으로 판단할t 좌우 편차 임계값 (픽셀 단위)

            if yaw_deviation > yaw_threshold:
                orientation_text = "Left"
                joint_pos[0] += ROTATION_SPEED 
            elif yaw_deviation < -yaw_threshold:
                orientation_text = "Right"
                joint_pos[0] -= ROTATION_SPEED
            
            # 2. Joint 2 & 3: 화면 3분할 상하 위치 추적
            face_center_y = int(nose_tip.y * camera_height)
            upper_bound = camera_height / 3
            lower_bound = camera_height * 2 / 3
            vertical_change = 0
            
            if face_center_y < upper_bound:
                joint_pos[1] -= VERTICAL_SPEED
                joint_pos[2] += VERTICAL_SPEED
                vertical_change = VERTICAL_SPEED 
            elif face_center_y > lower_bound:
                joint_pos[1] += VERTICAL_SPEED
                joint_pos[2] -= VERTICAL_SPEED
                vertical_change = -VERTICAL_SPEED

            # 3. Joint 4: Joint 2, 3의 상하 움직임을 보상
            joint_pos[3] -= vertical_change
            
            update_motors(joint_pos)
            cv2.putText(frame, f"Orientation: {orientation_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print("🛑 종료 및 포트 닫힘")
    cap.release()
    cv2.destroyAllWindows()
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
