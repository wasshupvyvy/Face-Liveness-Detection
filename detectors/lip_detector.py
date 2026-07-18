from scipy.spatial import distance as dist
import numpy as np

# lớp LipDetector sử dụng tỷ lệ khung hình môi để phát hiện mở miệng
class LipDetector:
    def __init__(self, threshold=0.3):
        self.threshold = threshold # Ngưỡng mở miệng

    def check(self, lip_points):
        # Lấy các điểm đặc trưng của môi trong/ngoài
        # Đơn giản hóa: dùng chiều cao / chiều rộng môi
        # lip_points shape: (N, 2)
        
        # Tính bao đóng (bounding box) của môi
        x, y, w, h = cv2.boundingRect(np.array(lip_points))
        aspect_ratio = float(h) / w
        
        # Nếu tỷ lệ cao/rộng lớn -> đang mở miệng
        return aspect_ratio > self.threshold, aspect_ratio