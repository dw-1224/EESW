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
P_GAIN_YAW = 0.0003   # 머리 회전에 따른 좌우 제어 게인
P_GAIN_PITCH = 0.0005  # 상하 센터링을 위한 비례 게인
P_GAIN_DEPTH = 0.00005 # 전후 거리 조절을 위한 게인 (속도 추가 감소)
DEPTH_SMOOTHING_ALPHA = 0.2 # 거리 제어의 떨림을 줄이기 위한 스무딩 값

MOTION_TOLERANCE = 20 # 움직임을 시작하기 위한 오차 임계값 (픽셀)

camera_width, camera_height = 640, 480
TARGET_FACE_WIDTH = int(camera_width * 0.6) # 목표 얼굴 너비 (카메라 너비의 60%)
FACE_WIDTH_TOLERANCE = int(TARGET_FACE_WIDTH * 0.05) # 목표 너비의 5% 허용 오차
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

# -------------------- Joint Position 변수 설정 -------------------- #
joint_pos = list(initial_joint_pos_rad)
smoothed_depth_error = 0.0 # 거리 제어 오차 스무딩을 위한 변수

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
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # --- 랜드마크에서 Bounding Box 계산 및 그리기 ---
            x_min = camera_width
            y_min = camera_height
            x_max = y_max = 0
            for landmark in landmarks:
                x = int(landmark.x * camera_width)
                y = int(landmark.y * camera_height)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # --- 제어 로직 시작 ---
            
            # 1. 랜드마크 추출 및 오차 계산
            nose_tip = landmarks[1]
            left_eye_inner = landmarks[226]
            right_eye_inner = landmarks[446]

            # Pitch control (상하): 화면 중앙을 기준으로 제어
            face_center_y = nose_tip.y * camera_height
            y_error = face_center_y - (camera_height / 2)

            # Depth control (전후): 얼굴 크기를 기준으로 제어
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            face_width_pixels = math.hypot(
                (left_cheek.x - right_cheek.x) * camera_width,
                (left_cheek.y - right_cheek.y) * camera_height
            )
            raw_depth_error = face_width_pixels - TARGET_FACE_WIDTH
            smoothed_depth_error = (DEPTH_SMOOTHING_ALPHA * raw_depth_error) + ((1.0 - DEPTH_SMOOTHING_ALPHA) * smoothed_depth_error)

            # Yaw control (좌우): 머리 회전 각도를 기준으로 제어
            eye_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
            yaw_deviation = (nose_tip.x - eye_center_x) * camera_width

            # --- 조인트별 제어 (각 축은 독립적으로 멈춤) ---

            # 1. Joint 1: Yaw (머리 회전으로 제어)
            # 사용자의 관찰에 따라 제어 방향을 반전시켰습니다.
            if abs(yaw_deviation) > MOTION_TOLERANCE:
                joint_pos[0] -= (yaw_deviation * P_GAIN_YAW)
            
            # 2. Joint 2 & 3: Pitch (상하 위치로 제어)
            if abs(y_error) > MOTION_TOLERANCE:
                pitch_adjustment = y_error * P_GAIN_PITCH
                joint_pos[1] += pitch_adjustment
                joint_pos[2] -= pitch_adjustment

            # 3. Joint 4: Depth (얼굴 크기로 제어)
            if abs(smoothed_depth_error) > FACE_WIDTH_TOLERANCE:
                depth_adjustment = smoothed_depth_error * P_GAIN_DEPTH
                joint_pos[3] += depth_adjustment
            
            update_motors(joint_pos)

            # --- 상태 정보 표시 ---
            cv2.putText(frame, f"Yaw_Input: {int(yaw_deviation)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Y_Err: {int(y_error)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Width_Err: {int(smoothed_depth_error)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print("🛑 종료. 초기 자세로 복귀합니다...")
    # 초기 자세로 이동
    for i, dxl_id in enumerate(DXL_IDS):
        packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, INITIAL_POSITIONS[i])
    
    # 모터가 초기 위치로 이동할 시간을 줍니다.
    print("ℹ️ 3초간 대기합니다...")
    time.sleep(3)

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    
    # 모든 모터의 토크를 비활성화합니다.
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    
    portHandler.closePort()
    print("✅ 초기 자세 복귀 완료 및 포트 닫힘.")