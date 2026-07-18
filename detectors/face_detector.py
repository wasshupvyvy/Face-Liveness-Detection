import cv2
import mediapipe as mp
import numpy as np

# lớp FaceDetector sử dụng MediaPipe để phát hiện khuôn mặt và trích xuất landmarks
class FaceDetector:
    # khởi tạo bộ phát hiện khuôn mặt MediaPipe
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh # Sử dụng Face Mesh của MediaPipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) # Khởi tạo FaceMesh với tham số

    # phát hiện khuôn mặt và trả về landmarks cùng kích thước khung hình
    def detect(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB
        results = self.face_mesh.process(rgb_image) # Xử lý ảnh để phát hiện khuôn mặt
        if results.multi_face_landmarks: 
            # Trả về landmarks của khuôn mặt đầu tiên
            return results.multi_face_landmarks[0], image.shape
        return None, None
    
    # tính bounding box từ landmarks
    def get_bbox(self, landmarks, frame_shape):
        h, w, _ = frame_shape # Lấy kích thước khung hình
        pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]) # Chuyển landmarks sang tọa độ pixel
        x, y, w_rect, h_rect = cv2.boundingRect(pts) # Tính bounding box
        return x, y, w_rect, h_rect # Trả về tọa độ và kích thước bounding box