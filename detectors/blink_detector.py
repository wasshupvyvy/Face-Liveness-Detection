from scipy.spatial import distance as dist

# lớp BlinkDetector sử dụng Eye Aspect Ratio (EAR) để phát hiện chớp mắt
class BlinkDetector:
    # khoi tao voi nguong EAR mac dinh la 0.25
    def __init__(self, threshold=0.25):
        self.threshold = threshold

    # Tính toán Eye Aspect Ratio (EAR)
    def _eye_aspect_ratio(self, eye):
        # eye là list các điểm tọa độ (x, y)
        A = dist.euclidean(eye[1], eye[5]) # Khoảng cách dọc 1
        B = dist.euclidean(eye[2], eye[4]) # Khoảng cách dọc 2
        C = dist.euclidean(eye[0], eye[3]) # Khoảng cách ngang
        return (A + B) / (2.0 * C)

    # Kiểm tra xem mắt có đang nhắm không dựa trên EAR
    def check(self, landmarks, w, h):
        # Lấy tọa độ mắt trái/phải từ landmarks MediaPipe
        # Index mắt trái: 362, 385, 387, 263, 373, 380
        # Index mắt phải: 33, 160, 158, 133, 153, 144
        
        def get_coords(indices):
            return [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]

        left_eye = get_coords([362, 385, 387, 263, 373, 380]) # mắt trái
        right_eye = get_coords([33, 160, 158, 133, 153, 144]) # mắt phải

        ear_left = self._eye_aspect_ratio(left_eye) # mắt trái
        ear_right = self._eye_aspect_ratio(right_eye)# mắt phải
        avg_ear = (ear_left + ear_right) / 2.0 # trung bình EAR

        return avg_ear < self.threshold # trả về True nếu mắt nhắm (EAR < ngưỡng)