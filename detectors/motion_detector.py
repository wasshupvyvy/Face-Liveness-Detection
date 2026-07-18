# detectors/motion_detector.py
import numpy as np

class MotionDetector:
    def __init__(self, max_history=30):
        self.history = []
        self.max_history = max_history # Lưu trữ 30 frame (~1 giây)

    def update(self, landmarks):
        # Lấy tọa độ 4 điểm neo quan trọng: Mũi(1), Mắt trái(33), Mắt phải(263), Cằm(152)
        # Các điểm này đại diện tốt nhất cho chuyển động đầu
        points = [landmarks.landmark[i] for i in [1, 33, 263, 152]]
        # Lấy tọa độ tương đối (trừ đi vị trí mũi) để loại bỏ việc di chuyển tịnh tiến cả người
        nose = landmarks.landmark[1]
        rel_points = []
        for p in points: # Tính tọa độ tương đối
            rel_points.append((p.x - nose.x, p.y - nose.y)) # Chỉ lấy x,y, bỏ z
        
        self.history.append(rel_points) # Thêm vào lịch sử
        if len(self.history) > self.max_history: # Giữ kích thước lịch sử không vượt quá max_history
            self.history.pop(0) # Xoá frame cũ nhất

        return self._calculate_variance() # Trả về chỉ số chuyển động

    def _calculate_variance(self): 
        if len(self.history) < 10: 
            return 99.0 # Chưa đủ dữ liệu thì mặc định là sống động (để không chặn nhầm lúc đầu)
        
        # Chuyển list sang numpy array để tính toán ma trận
        data = np.array(self.history) # Shape: (frames, num_points, 2)
        
        # Tính độ lệch chuẩn (Standard Deviation) dọc theo trục thời gian (axis 0)
        # Ý nghĩa: Các điểm neo này dao động bao nhiêu so với vị trí trung bình của chính nó?
        # Ảnh tĩnh: std cực thấp (gần 0). Người thật: std cao hơn do hô hấp/cơ mặt.
        std_dev = np.std(data, axis=0) 
        
        # Lấy trung bình cộng của các độ lệch chuẩn và nhân hệ số phóng đại
        avg_motion = np.mean(std_dev) * 1000 
        return avg_motion
    
    def reset(self):
        self.history = []
