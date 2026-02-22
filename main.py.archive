import cv2
import mediapipe as mp
import numpy as np
import time

# FaceLandmarkerの初期化
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# モデルのダウンロードが必要: face_landmarker_v2_with_blendshapes.task
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# EAR（Eye Aspect Ratio）の計算関数
def calculate_ear(eye_landmarks):
    # 垂直距離の計算
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    
    # 水平距離の計算
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    # EARの計算
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# 瞬き検出のメイン処理
cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.2  # 瞬き判定の閾値
CONSECUTIVE_FRAMES = 2  # 連続フレーム数

blink_counter = 0
total_blinks = 0
frame_counter = 0
blink_interval = 0
last_time_blinked = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # ランドマーク検出
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            
            # 左目のランドマークインデックス（MediaPipe仕様）
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            # 右目のランドマークインデックス
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            # 目のランドマーク座標を取得
            left_eye = [(face_landmarks[i].x, face_landmarks[i].y) for i in left_eye_indices]
            right_eye = [(face_landmarks[i].x, face_landmarks[i].y) for i in right_eye_indices]
            
            # 両目のEARを計算
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # 瞬き判定
            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSECUTIVE_FRAMES:
                    total_blinks += 1
                    print(f"瞬きを検出！ (合計: {total_blinks}回)")
                    now = time.time()
                    blink_interval = now - last_time_blinked
                    last_time_blinked = now
                blink_counter = 0
            
            # 画面に表示
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"BlinkInterval: {blink_interval:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Blink Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# def main():
#     print("Hello from blinkmore-zero!")
#
#
# if __name__ == "__main__":
#     main()
