import cv2
import numpy as np
import time
import random
from detectors.face_detector import FaceDetector
from detectors.emotion_detector import EmotionDetector
from detectors.motion_detector import MotionDetector

STATE_WAITING = 0      
STATE_ANALYZING = 1    
STATE_CHALLENGE = 2    
STATE_RESULT = 3       

STATIC_THRESHOLD = 1.5 

# Vẽ bảng thông tin trên khung hình
def draw_dashboard(frame, emotion, blink_count, motion_score, state_text, state_color):
    """Bảng thông tin"""
    h_frame, w_frame, _ = frame.shape
    
    # 1. Vẽ nền bảng thông tin (Góc trái trên)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    
    # 2. Vẽ nền thanh hướng dẫn (Góc dưới)
    cv2.rectangle(overlay, (0, h_frame - 40), (w_frame, h_frame), (0, 0, 0), -1)
    
    # Áp dụng độ trong suốt
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    #HIỂN THỊ THÔNG SỐ DASHBOARD
    cv2.putText(frame, "--- DASHBOARD ---", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Cảm xúc
    e_color = (255, 255, 255)
    if "SMILE" in emotion: e_color = (0, 255, 0)
    elif "SURPRISE" in emotion: e_color = (0, 255, 255)
    elif "BLINK" in emotion: e_color = (100, 100, 255)
    cv2.putText(frame, f"Emotion: {emotion}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, e_color, 2)
    
    # Số lần chớp mắt
    cv2.putText(frame, f"Blinks: {blink_count}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Chỉ số chuyển động
    m_color = (0, 255, 0) if motion_score > STATIC_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"Motion: {motion_score:.2f}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 1)

    # --- HIỂN THỊ HƯỚNG DẪN THOÁT (QUAN TRỌNG) ---
    cv2.putText(frame, "PRESS 'Q' TO QUIT", (w_frame // 2 - 100, h_frame - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Trạng thái hệ thống (Pass/Fail)
    if state_text:
        cv2.putText(frame, state_text, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

# Chương trình chính
def main():
    face_det = FaceDetector() 
    emotion_det = EmotionDetector() 
    motion_det = MotionDetector()

    cap = cv2.VideoCapture(0) # Mở camera mặc định
    
    current_state = STATE_WAITING # Trạng thái ban đầu
    
    # Biến thách thức
    challenge_type = ""
    challenge_timer = 0
    CHALLENGE_LIMIT = 5.0
    # Kết quả
    result_text = ""
    result_color = (0,0,0)
    result_timer = 0
    # Đếm chớp mắt
    blink_count = 0
    is_eye_closed = False 

    print("Hệ thống đang chạy.\n Bấm vào cửa sổ camera và nhấn 'Q' để thoát.")
    # Vòng lặp chính
    while True:
        ret, frame = cap.read() # Đọc khung hình từ camera
        if not ret: break # Nếu không đọc được thì thoát
        
        frame = cv2.flip(frame, 1) # Lật khung hình ngang
        h, w, _ = frame.shape # Kích thước khung hình

        # 1. Phát hiện mặt
        landmarks, shape = face_det.detect(frame)
        
        motion_score = 0.0
        current_emotion = "No Face"
        # Xử lý theo trạng thái
        if not landmarks: # Không phát hiện mặt
            current_state = STATE_WAITING # Quay về trạng thái chờ
            motion_det.reset()
            is_eye_closed = False # Đặt lại trạng thái chớp mắt
            cv2.putText(frame, "Waiting for face...", (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1) # Thông báo chờ mặt
        else:
            fx, fy, fw, fh = face_det.get_bbox(landmarks, shape) # Lấy hộp giới hạn mặt
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2) # Vẽ hộp giới hạn mặt

            # Nhận diện
            current_emotion, _ = emotion_det.detect_state(landmarks, w, h)
            
            # Đếm chớp mắt
            if "BLINKING" in current_emotion or "CLOSED" in current_emotion:
                if not is_eye_closed:
                    is_eye_closed = True 
            else:
                if is_eye_closed:
                    blink_count += 1
                    is_eye_closed = False

            # Cập nhật Motion
            motion_score = motion_det.update(landmarks)

            #trạng thái
            if current_state == STATE_WAITING:
                if len(motion_det.history) >= 20: # Đủ dữ liệu để phân tích
                    current_state = STATE_ANALYZING # Chuyển sang trạng thái phân tích

            elif current_state == STATE_ANALYZING: # Phân tích chuyển động
                if motion_score < STATIC_THRESHOLD: # Phát hiện tĩnh
                    current_state = STATE_RESULT # Chuyển sang trạng thái kết quả
                    result_text = "FAKE: STATIC PHOTO"
                    result_color = (0, 0, 255)
                    result_timer = time.time()
                else:
                    challenges = ["SMILE", "SURPRISE", "BLINK"]
                    challenge_type = random.choice(challenges)
                    challenge_timer = time.time()
                    current_state = STATE_CHALLENGE

            elif current_state == STATE_CHALLENGE:
                elapsed = time.time() - challenge_timer
                time_left = CHALLENGE_LIMIT - elapsed
                # Hiển thị thách thức
                cv2.putText(frame, f"PLEASE: {challenge_type}", (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"Time: {time_left:.1f}s", (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                # Kiểm tra kết quả thách thức
                if motion_score < STATIC_THRESHOLD:
                     current_state = STATE_RESULT
                     result_text = "FAKE: KE GIA MAO"
                     result_color = (0, 0, 255)
                     result_timer = time.time()
                # Kiểm tra thách thức
                passed = False
                if challenge_type == "SMILE" and "SMILING" in current_emotion: passed = True
                elif challenge_type == "SURPRISE" and "SURPRISED" in current_emotion: passed = True
                elif challenge_type == "BLINK" and "BLINKING" in current_emotion: passed = True
                # Chuyển sang trạng thái kết quả nếu vượt qua thách thức
                if passed:
                    current_state = STATE_RESULT
                    result_text = "ACCESS GRANTED"
                    result_color = (0, 255, 0)
                    result_timer = time.time()
                # Kiểm tra hết thời gian
                if time_left <= 0: # Hết thời gian
                    current_state = STATE_RESULT # Chuyển sang trạng thái kết quả
                    result_text = "FAILED: TIME OUT" # Thông báo thất bại
                    result_color = (0, 0, 255)
                    result_timer = time.time() # Đặt thời gian kết quả
            # Hiển thị kết quả
            elif current_state == STATE_RESULT: 
                cv2.putText(frame, result_text, (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2) # Hiển thị kết quả
                if time.time() - result_timer > 3.0:
                    current_state = STATE_WAITING # Quay về trạng thái chờ
                    motion_det.reset()
                    blink_count = 0

        # Vẽ Dashboard
        draw_dashboard(frame, current_emotion, blink_count, motion_score, "", (0,0,0))

        cv2.imshow("Face Liveness System", frame)
        
        # Xử lý thoát
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Đã nhận lệnh thoát (Q).")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()