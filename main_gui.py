import sys
import os
import cv2
import time
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QGridLayout, QSizePolicy, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

# --- CẤU HÌNH HIỂN THỊ ---
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

# --- IMPORT CÁC MODULE XỬ LÝ AI ---
try:
    from detectors.face_detector import FaceDetector
    from detectors.emotion_detector import EmotionDetector
    from detectors.motion_detector import MotionDetector
except ImportError:
    # Tạo các lớp giả lập để test giao diện nếu thiếu file backend
    class FaceDetector: 
        def detect(self, img): return [], None
        def get_bbox(self, l, s): return 0,0,0,0
    class EmotionDetector:
        def detect_state(self, l, w, h): return "No Face", 0
    class MotionDetector:
        def update(self, l): return 0.0
        def reset(self): pass
        self.history = []

# --- ĐỊNH NGHĨA CÁC TRẠNG THÁI (STATE MACHINE) ---
STATE_IDLE = -1          # Trạng thái nghỉ (Chưa bật camera)
STATE_WAITING = 0        # Trạng thái chờ: Đang quét tìm mặt
STATE_ANALYZING = 1      # Trạng thái phân tích: Kiểm tra ảnh tĩnh/động
STATE_CHALLENGE = 2      # Trạng thái thử thách: Yêu cầu người dùng hành động
STATE_RESULT = 3         # Trạng thái kết quả: Hiển thị Đạt/Không đạt

# Các ngưỡng số liệu kỹ thuật
STATIC_THRESHOLD = 1.5   # Ngưỡng điểm chuyển động (Dưới mức này coi là ảnh tĩnh)
CHALLENGE_LIMIT = 5.0    # Thời gian tối đa để thực hiện thử thách (giây)

# --- LUỒNG XỬ LÝ AI ---
class AIWorker(QThread):
    # Gửi hình ảnh lên giao diện
    frame_update = Signal(QImage)
    # Gửi các thông số lên giao diện
    stats_update = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.is_running = True      
        self.camera_on = False      
        self.reset_logic()          

    def reset_logic(self):
        """Hàm đặt lại toàn bộ các biến logic về trạng thái ban đầu"""
        self.current_state = STATE_WAITING
        self.challenge_type = ""
        self.result_text = ""
        self.result_color = "#FFFFFF"
        self.blink_count = 0
        self.is_eye_closed = False
        self.motion_score = 0.0
        self.need_reset_detector = True # Reset bộ nhớ detector
        self.challenge_timer = 0
        self.result_timer = 0

    def set_camera(self, on):
        """Hàm nhận lệnh Bật/Tắt từ giao diện chính"""
        self.camera_on = on
        if not on:
            self.current_state = STATE_IDLE
        else:
            self.current_state = STATE_WAITING
            self.need_reset_detector = True

    def run(self):
        """Hàm chạy chính của luồng"""
        # Khởi tạo các mô hình AI
        face_det = FaceDetector()
        emotion_det = EmotionDetector()
        motion_det = MotionDetector()
        cap = None

        while self.is_running:
            # === TRƯỜNG HỢP 1: CAMERA ĐANG TẮT ===
            if not self.camera_on:
                # Nếu camera đang mở
                if cap is not None:
                    cap.release()
                    cap = None
                
                off_frame = QImage(640, 480, QImage.Format_RGB888)
                off_frame.fill(QColor(224, 247, 250)) 
                self.frame_update.emit(off_frame)
                
                self.stats_update.emit({
                    "emotion": "OFF", "blink": 0, "motion": 0.0,
                    "instruction": "PRESS 'START' TO BEGIN",
                    "status_color": "#B2EBF2", "text_color": "#006064"
                })
                time.sleep(0.1)
                continue

            # === TRƯỜNG HỢP 2: CAMERA ĐANG BẬT ===
            # Nếu chưa kết nối camera thì kết nối
            if cap is None:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.reset_logic()

            # Đọc khung hình từ camera
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue

            # Lật ngược ảnh (hiệu ứng gương)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Reset các bộ detector nếu có yêu cầu
            if self.need_reset_detector:
                motion_det.reset(); self.blink_count = 0; self.need_reset_detector = False

            # Khởi tạo các biến hiển thị mặc định
            instruction_text = "..."
            status_color = "#FFF"
            text_color = "#333"
            current_emotion = "--"

            # 1. Phát hiện khuôn mặt
            landmarks, shape = face_det.detect(frame)

            if not landmarks:
                # Nếu không thấy mặt -> Quay về trạng thái chờ
                self.current_state = STATE_WAITING
                instruction_text = "FACE NOT FOUND"
                status_color = "#FFF9C4" # Vàng nhạt cảnh báo
                text_color = "#F57F17"
                motion_det.reset()
            else:
                # Lấy tọa độ hộp bao quanh mặt (Bounding Box)
                fx, fy, fw, fh = face_det.get_bbox(landmarks, shape)
                self.draw_corners(frame, fx, fy, fw, fh)
                
                # 2. Nhận diện cảm xúc & Trạng thái mắt
                current_emotion, _ = emotion_det.detect_state(landmarks, w, h)
                
                # Logic đếm số lần chớp mắt
                if "BLINKING" in current_emotion or "CLOSED" in current_emotion:
                    if not self.is_eye_closed: self.is_eye_closed = True 
                else:
                    if self.is_eye_closed: self.blink_count += 1; self.is_eye_closed = False
                
                # 3. Tính điểm chuyển động (Liveness Score)
                self.motion_score = motion_det.update(landmarks)
                
                # Giai đoạn: CHỜ ỔN ĐỊNH
                if self.current_state == STATE_WAITING:
                    instruction_text = "SCANNING FACE..."
                    status_color = "#E1F5FE" 
                    text_color = "#0277BD"
                    # Nếu thu thập đủ 20 frame thì chuyển sang phân tích
                    if len(motion_det.history) >= 20: 
                        self.current_state = STATE_ANALYZING

                # Giai đoạn: PHÂN TÍCH ĐỘ TĨNH
                elif self.current_state == STATE_ANALYZING:
                    instruction_text = "ANALYZING LIVENESS..."
                    # Nếu điểm chuyển động thấp hơn ngưỡng -> Nghi ngờ ảnh tĩnh
                    if self.motion_score < STATIC_THRESHOLD:
                        self.current_state = STATE_RESULT
                        self.result_text = "WARNING: STATIC IMAGE"
                        self.result_color = "#FFEBEE"; self.text_res_color = "#C62828"
                        self.result_timer = time.time()
                    else:
                        # Nếu chuyển động tốt -> TỰ ĐỘNG chọn thử thách ngẫu nhiên
                        challenges = ["SMILE", "SURPRISE", "BLINK"]
                        self.challenge_type = random.choice(challenges)
                        self.challenge_timer = time.time()
                        self.current_state = STATE_CHALLENGE

                # Giai đoạn: THỰC HIỆN THỬ THÁCH
                elif self.current_state == STATE_CHALLENGE:
                    elapsed = time.time() - self.challenge_timer
                    time_left = CHALLENGE_LIMIT - elapsed
                    
                    eng_map = {
                        "SMILE": "PLEASE SMILE", 
                        "SURPRISE": "SHOW SURPRISE", 
                        "BLINK": "BLINK EYES"
                    }
                    req_text = eng_map.get(self.challenge_type, self.challenge_type)
                    
                    instruction_text = f"ACTION: {req_text} ({time_left:.1f}s)"
                    status_color = "#FFF3E0" # Màu cam nhạt
                    text_color = "#EF6C00"

                    # Kiểm tra liên tục
                    if self.motion_score < STATIC_THRESHOLD - 0.5:
                         self.current_state = STATE_RESULT
                         self.result_text = "SPOOF DETECTED"
                         self.result_color = "#FFEBEE"; self.text_res_color = "#C62828"
                         self.result_timer = time.time()

                    # Kiểm tra hành động người dùng
                    passed = False
                    if self.challenge_type == "SMILE" and "SMILING" in current_emotion: passed = True
                    elif self.challenge_type == "SURPRISE" and "SURPRISED" in current_emotion: passed = True
                    elif self.challenge_type == "BLINK" and "BLINKING" in current_emotion: passed = True

                    # Xử lý kết quả thử thách
                    if passed:
                        self.current_state = STATE_RESULT
                        self.result_text = "ACCESS GRANTED"
                        self.result_color = "#E8F5E9"; self.text_res_color = "#2E7D32"
                        self.result_timer = time.time()
                    elif time_left <= 0:
                        self.current_state = STATE_RESULT
                        self.result_text = "FAILED: TIMEOUT"
                        self.result_color = "#FFEBEE"; self.text_res_color = "#C62828"
                        self.result_timer = time.time()

                # Giai đoạn: HIỂN THỊ KẾT QUẢ
                elif self.current_state == STATE_RESULT:
                    instruction_text = self.result_text
                    status_color = self.result_color
                    text_color = getattr(self, 'text_res_color', "#000")
                    
                    if time.time() - self.result_timer > 3.0:
                        self.current_state = STATE_WAITING
                        motion_det.reset()
                        self.blink_count = 0

            # Chuyển đổi màu từ OpenCV (BGR) sang Qt (RGB) để hiển thị đúng màu
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, w, h, w*3, QImage.Format_RGB888)
            self.frame_update.emit(qt_image)
            
            # Gửi dữ liệu thống kê về giao diện
            self.stats_update.emit({
                "emotion": current_emotion, "blink": self.blink_count,
                "motion": self.motion_score, "instruction": instruction_text,
                "status_color": status_color, "text_color": text_color
            })
            # Giới hạn tốc độ khung hình (~30 FPS) để giảm tải
            time.sleep(0.03)

    def draw_corners(self, img, x, y, w, h):
        """Hàm vẽ 4 góc bao quanh khuôn mặt"""
        color = (255, 191, 0) 
        t = 2; l = 25
        # Góc trên trái
        cv2.line(img, (x, y), (x + l, y), color, t); cv2.line(img, (x, y), (x, y + l), color, t)
        # Góc trên phải
        cv2.line(img, (x + w, y), (x + w - l, y), color, t); cv2.line(img, (x + w, y), (x + w, y + l), color, t)
        # Góc dưới trái
        cv2.line(img, (x, y + h), (x + l, y + h), color, t); cv2.line(img, (x, y + h), (x, y + h - l), color, t)
        # Góc dưới phải
        cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, t); cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, t)

    def stop_worker(self):
        """Hàm dừng luồng an toàn"""
        self.is_running = False
        self.wait()

# --- LỚP GIAO DIỆN CHÍNH (GUI) ---
class FaceIDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Liveness Detection")
        self.resize(1100, 680)
        
        # --- CẤU HÌNH GIAO DIỆN (CSS) ---
        self.setStyleSheet("""
            QMainWindow { background-color: #E0F7FA; }
            QLabel { font-family: 'Segoe UI', Arial; color: #006064; }
            QFrame#ScoreBoard { 
                background-color: #FFFFFF; 
                border: 3px solid #00BCD4; 
                border-radius: 12px; 
            }
            QFrame#ControlPanel { background-color: transparent; }
        """)

        # Widget trung tâm chứa layout chính
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # === 1. PHẦN CAMERA ===
        video_wrapper = QFrame()
        video_wrapper.setStyleSheet("background: #FFF; border: 1px solid #B2EBF2; border-radius: 10px;")
        v_layout = QVBoxLayout(video_wrapper)
        v_layout.setContentsMargins(5,5,5,5)
        
        self.lbl_video = QLabel("CAMERA OFF")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_video.setScaledContents(True)
        self.lbl_video.setStyleSheet("background-color: #000; border-radius: 6px; color: #888;")
        v_layout.addWidget(self.lbl_video)

        self.lbl_instruction = QLabel("SYSTEM OFFLINE")
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_instruction.setFixedHeight(60)
        self.lbl_instruction.setStyleSheet("background: #B2EBF2; color: #006064; border-radius: 6px;")
        v_layout.addWidget(self.lbl_instruction)

        main_layout.addWidget(video_wrapper, stretch=65)

        # === 2. THANH BÊN ===
        sidebar = QFrame()
        sidebar.setFixedWidth(340)
        s_layout = QVBoxLayout(sidebar)
        s_layout.setSpacing(15)
        s_layout.setContentsMargins(0,0,0,0)

        # --- A. BẢNG ĐIỂM ---
        score_frame = QFrame()
        score_frame.setObjectName("ScoreBoard")
        sf_layout = QVBoxLayout(score_frame)
        
        lbl_title = QLabel("SCORE BOARD")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #0097A7; margin-bottom: 5px;")
        sf_layout.addWidget(lbl_title)
        
        sf_grid = QGridLayout()
        sf_grid.setVerticalSpacing(15)

        # 1. Điểm chuyển động
        sf_grid.addWidget(QLabel("Motion Score:"), 0, 0)
        self.lbl_motion = QLabel("0.00")
        self.lbl_motion.setAlignment(Qt.AlignRight)
        self.lbl_motion.setFont(QFont("Segoe UI", 18, QFont.Bold))
        sf_grid.addWidget(self.lbl_motion, 0, 1)

        # 2. Số lần chớp mắt
        sf_grid.addWidget(QLabel("Blink Count:"), 1, 0)
        self.lbl_blink = QLabel("0")
        self.lbl_blink.setAlignment(Qt.AlignRight)
        self.lbl_blink.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.lbl_blink.setStyleSheet("color: #0277BD;") 
        sf_grid.addWidget(self.lbl_blink, 1, 1)

        # 3. Cảm xúc
        sf_grid.addWidget(QLabel("Emotion:"), 2, 0)
        self.lbl_emo = QLabel("--")
        self.lbl_emo.setAlignment(Qt.AlignRight)
        self.lbl_emo.setFont(QFont("Segoe UI", 14, QFont.Bold))
        sf_grid.addWidget(self.lbl_emo, 2, 1)

        sf_layout.addLayout(sf_grid)
        s_layout.addWidget(score_frame)

        # --- B. BẢNG ĐIỀU KHIỂN ---
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("ControlPanel")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        
        lbl_ctrl = QLabel("CONTROLS")
        lbl_ctrl.setAlignment(Qt.AlignCenter)
        lbl_ctrl.setStyleSheet("font-weight: bold; color: #00838F;")
        ctrl_layout.addWidget(lbl_ctrl)

        # Nút Start
        self.btn_start = QPushButton("START CAMERA")
        self.btn_start.setMinimumHeight(55)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet(self.btn_style("#00C853"))
        self.btn_start.clicked.connect(self.click_start)
        ctrl_layout.addWidget(self.btn_start)

        # Nút Stop
        self.btn_stop = QPushButton("STOP CAMERA")
        self.btn_stop.setMinimumHeight(55)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet(self.btn_style("#D50000"))
        self.btn_stop.clicked.connect(self.click_stop)
        self.btn_stop.setVisible(False)
        ctrl_layout.addWidget(self.btn_stop)

        # Nút Reset
        self.btn_reset = QPushButton("RESET SYSTEM")
        self.btn_reset.setMinimumHeight(55)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet(self.btn_style("#0288D1"))
        self.btn_reset.clicked.connect(self.click_reset)
        ctrl_layout.addWidget(self.btn_reset)

        s_layout.addWidget(ctrl_frame)
        s_layout.addStretch()

        # --- C. Help
        help_layout = QHBoxLayout()
        help_layout.addStretch()
        
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(45, 45)
        self.btn_help.setFont(QFont("Arial", 16, QFont.Bold))
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.setToolTip("User Guide")
        self.btn_help.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; color: white; border-radius: 22px;
                border: 2px solid #FFF;
            }
            QPushButton:hover { background-color: #F57C00; }
        """)
        self.btn_help.clicked.connect(self.show_help_dialog)
        help_layout.addWidget(self.btn_help)
        
        s_layout.addLayout(help_layout)

        main_layout.addWidget(sidebar, stretch=35)

        # === KẾT NỐI VỚI LUỒNG AI ===
        self.worker = AIWorker()
        self.worker.frame_update.connect(self.update_video)
        self.worker.stats_update.connect(self.update_stats)
        self.worker.start()

        QTimer.singleShot(500, self.show_help_dialog)

    def btn_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color}; color: white; 
                border-radius: 8px; font-weight: bold; font-size: 14px;
            }}
            QPushButton:hover {{ filter: brightness(110%); }}
            QPushButton:pressed {{ margin-top: 2px; }}
        """

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN NÚT BẤM ---
    def click_start(self):
        """Khi bấm START: Bật camera, ẩn Start, hiện Stop"""
        self.worker.set_camera(True)
        self.btn_start.setVisible(False)
        self.btn_stop.setVisible(True)
        self.btn_reset.setEnabled(False) # Khóa nút Reset khi đang chạy
        self.btn_reset.setStyleSheet(self.btn_style("#B0BEC5"))

    def click_stop(self):
        """Khi bấm STOP: Tắt camera, ẩn Stop, hiện Start"""
        self.worker.set_camera(False)
        self.btn_stop.setVisible(False)
        self.btn_start.setVisible(True)
        self.btn_reset.setEnabled(True) # Mở khóa nút Reset
        self.btn_reset.setStyleSheet(self.btn_style("#0288D1"))

    def click_reset(self):
        """Khi bấm RESET: Đặt lại toàn bộ logic"""
        self.worker.reset_logic()
        self.lbl_instruction.setText("SYSTEM RESET DONE")

    def show_help_dialog(self):
        """Hiển thị hộp thoại Hướng dẫn sử dụng"""
        msg = QMessageBox(self)
        msg.setWindowTitle("User Guide")
        msg.setTextFormat(Qt.RichText)
        msg.setText("""
        <h3 style='color:#0097A7'>SYSTEM INSTRUCTIONS</h3>
        <p>This system performs liveness detection automatically.</p>
        <ol>
            <li>Press <b>[START CAMERA]</b> to begin.</li>
            <li>Keep your face steady within the frame.</li>
            <li>The system will randomly request an action:
                <ul>
                    <li><b>Smile</b></li>
                    <li><b>Show Surprise</b></li>
                    <li><b>Blink Eyes</b></li>
                </ul>
            </li>
            <li>Perform the action within 5 seconds.</li>
        </ol>
        <hr>
        <p><i>Use <b>[RESET SYSTEM]</b> to clear previous results.</i></p>
        """)
        msg.exec()

    # --- CẬP NHẬT GIAO DIỆN TỪ TÍN HIỆU CỦA AI ---
    @Slot(QImage)
    def update_video(self, img):
        """Cập nhật hình ảnh camera lên Label"""
        self.lbl_video.setPixmap(QPixmap.fromImage(img).scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(dict)
    def update_stats(self, s):
        """Cập nhật các số liệu lên Bảng Điểm"""
        self.lbl_motion.setText(f"{s['motion']:.2f}")
        self.lbl_blink.setText(str(s['blink']))
        self.lbl_emo.setText(s['emotion'])
        
        # Đổi màu điểm Motion: Đỏ (Cảnh báo) hoặc Xanh lá (An toàn)
        if s['motion'] < STATIC_THRESHOLD:
            self.lbl_motion.setStyleSheet("color: #D32F2F;") 
        else:
            self.lbl_motion.setStyleSheet("color: #388E3C;") 

        # Cập nhật thanh hướng dẫn bên dưới Camera
        self.lbl_instruction.setText(s['instruction'])
        self.lbl_instruction.setStyleSheet(f"background: {s['status_color']}; color: {s['text_color']}; border-radius: 6px;")

    def closeEvent(self, event):
        """Sự kiện khi tắt cửa sổ ứng dụng: Dừng luồng an toàn"""
        self.worker.stop_worker()
        event.accept()

# --- KHỞI CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceIDApp()
    win.show()
    sys.exit(app.exec())