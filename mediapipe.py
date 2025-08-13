#!/usr/bin/env python3
# 필요한 라이브러리들을 가져옵니다.
import cv2
import math
import time
from dynamixel_sdk import *
import mediapipe as mp

# -------------------- Dynamixel Setup (다이나믹셀 모터 설정) -------------------- #
DEVICENAME = 'COM7'         # 다이나믹셀이 연결된 COM 포트 이름 (Windows 기준)
BAUDRATE = 1000000          # 통신 속도
PROTOCOL_VERSION = 1.0      # 통신 프로토콜 버전 (AX-12A는 1.0 사용)

# 모터의 컨트롤 테이블 주소 정의
ADDR_TORQUE_ENABLE = 24     # 토크(모터 힘) ON/OFF 제어 주소
ADDR_GOAL_POSITION = 30     # 목표 위치 제어 주소
ADDR_MOVING_SPEED = 32      # 동작 속도 제어 주소

# 제어할 다이나믹셀 모터들의 ID 목록
DXL_IDS = [1, 2, 3, 4]
TORQUE_ENABLE = 1           # 토크를 켜기 위한 값 (1)

def radians_to_position(rad, scale=1.0):
    """
    라디안 값을 다이나믹셀 위치값(0~1023)으로 변환합니다. (수정된 최종 함수)
    - rad: 목표 각도(라디안)
    - scale: 최대 이동 범위 조절 (1.0 = 100%, 0.5 = 50%)
    """
    center_position = 512
    max_deviation = 511
    
    # 라디안 값을 -1.0 ~ +1.0 범위로 정규화하고 scale 적용
    scaled_rad = (rad / math.pi) * scale
    
    # 최종 위치 계산 (중앙 위치 + 변화량)
    goal_position = center_position + (scaled_rad * max_deviation)
    
    # 최종 위치값이 0~1023 범위를 벗어나지 않도록 제한
    return max(0, min(1023, int(goal_position)))

# -------------------- Port/Packet Handler Initialization (통신 포트 및 패킷 핸들러 초기화) -------------------- #
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("❌ 포트 열기 실패"); exit()
if not portHandler.setBaudRate(BAUDRATE):
    print("❌ 보율 설정 실패"); exit()
print("✅ 포트 연결 완료")

# --- 초기화 시 토크와 속도를 함께 설정 ---
DEFAULT_SPEED = 50  # 숫자가 클수록 빠름 (1~1023), 0은 최대 속도
for dxl_id in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_MOVING_SPEED, DEFAULT_SPEED)
print(f"✅ 모든 모터의 기본 속도가 {DEFAULT_SPEED}로 설정되었습니다.")
# ----------------------------------------------------

# -------------------- Face Tracker (얼굴 추적기 설정) -------------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패"); exit()

camera_width, camera_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

# -------------------- Joint Position (관절 위치 변수 설정) -------------------- #
joint_pos = [0.0, 0.0, 0.0, 0.0]  # 4개 관절의 초기 목표 각도 (라디안)
initial_face_height = None          # 처음 인식된 얼굴 크기를 저장할 변수

def update_motors(pos_rad):
    """라디안 값 리스트를 받아 모든 모터를 해당 위치로 이동시키는 함수"""
    for i, rad in enumerate(pos_rad):
        dxl_pos = radians_to_position(rad)
        packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_GOAL_POSITION, dxl_pos)

# -------------------- Main Loop (메인 실행 루프) -------------------- #
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 입력 실패"); break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
 
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 제어에 필요한 랜드마크 추출
            nose = landmarks[1]
            left_eye_top = landmarks[159]
            chin = landmarks[152]
            left_eye_l_corner = landmarks[33]
            left_eye_r_corner = landmarks[133]
            left_pupil = landmarks[473]

            face_center_y = int(nose.y * camera_height)
            center_y = camera_height // 2

            # --- 조인트별 추적 계산 (요청사항 반영된 최종 로직) ---

            # 1. Joint 1: Yaw (좌우 회전) 계산
            yaw_proxy = nose.x - chin.x
            YAW_THRESHOLD = 0.01  # 정면으로 판단할 임계값
            if abs(yaw_proxy) < YAW_THRESHOLD:
                joint_pos[0] *= 0.9  # 정면일 경우 서서히 중앙(0)으로 복귀
            else:
                # 얼굴 방향에 따라 회전 (민감도: 0.1)
                joint_pos[0] = max(-math.pi, min(math.pi, joint_pos[0] - yaw_proxy * 0.1))

            # 2. Joint 2: 거리 계산 (처음 인식된 얼굴 크기 기준)
            current_height = abs(chin.y - left_eye_top.y) * camera_height
            if initial_face_height is None:
                initial_face_height = current_height # 첫 프레임의 얼굴 크기를 기준점으로 저장
            # 기준점 대비 현재 얼굴 크기 차이로 전후 이동량 계산 (민감도: 0.01)
            dz = (initial_face_height - current_height) * 0.01

            # 3. Joint 3: 상하 이동
            dy = (face_center_y - center_y) * 0.005

            # 4. Joint 4: 눈동자 방향(Gaze) 추적
            eye_center_x = (left_eye_l_corner.x + left_eye_r_corner.x) / 2
            # 눈동자 위치로 시선 방향 추정 (민감도: 5.0)
            gaze_direction = (left_pupil.x - eye_center_x) * 5.0

            # --- 계산된 값으로 최종 joint_pos 업데이트 ---
            joint_pos[1] = max(-1.5, min(1.5, joint_pos[1] + dz))
            joint_pos[2] = max(-1.5, min(1.5, joint_pos[2] + dy))
            joint_pos[3] = max(-1.0, min(1.0, joint_pos[3] + gaze_direction))

            update_motors(joint_pos)
            cv2.putText(frame, f"Face Tracked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 얼굴이 감지되지 않으면, 다음 감지를 위해 초기 얼굴 크기 리셋
            initial_face_height = None
            cv2.putText(frame, f"No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # --- 프로그램 종료 시 정리 작업 ---
    print("🛑 종료 및 포트 닫힘")
    cap.release()
    cv2.destroyAllWindows()
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()