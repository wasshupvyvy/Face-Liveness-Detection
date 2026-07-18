from scipy.spatial import distance as dist
import numpy as np
from utils.landmarks import LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER

class EmotionDetector:
    def __init__(self):
        # Các ngưỡng (Threshold) có thể cần tinh chỉnh tùy camera
        self.BLINK_THRESH = 0.18      # Dưới mức này là nhắm mắt
        self.SMILE_THRESH = 0.42    # Mở rộng miệng bề ngang hoặc tỉ lệ khóe miệng
        self.SURPRISE_THRESH = 0.5    # Mở miệng rộng theo chiều dọc (há hốc)
        self.ANGRY_THRESH = 0.25    # Khoảng cách lông mày co lại

    def _get_aspect_ratio(self, points):
        # Tính toán tỉ lệ mở của mắt (EAR)
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])
        C = dist.euclidean(points[0], points[3])
        return (A + B) / (2.0 * C)

    def _get_mouth_ratio(self, landmarks, w, h):
        # Lấy tọa độ 4 điểm môi: Trái(61), Phải(291), Trên(0), Dưới(17)
        p61 = self._to_coords(landmarks.landmark[61], w, h)
        p291 = self._to_coords(landmarks.landmark[291], w, h)
        p0 = self._to_coords(landmarks.landmark[0], w, h)
        p17 = self._to_coords(landmarks.landmark[17], w, h)

        width = dist.euclidean(p61, p291)   # Chiều rộng miệng
        height = dist.euclidean(p0, p17)    # Chiều cao miệng
        
        # MAR (Mouth Aspect Ratio)
        if width == 0: return 0
        return height / width

    def _get_brow_eye_dist(self, landmarks, w, h):
        # Tính khoảng cách trung bình từ lông mày xuống mắt (cho Ngạc nhiên/Tức giận)
        # Lấy điểm giữa lông mày và điểm giữa mắt tương ứng
        #Lông mày 105 (trái), Mắt 373 (trái trên)
        brow_pt = self._to_coords(landmarks.landmark[105], w, h)
        eye_pt = self._to_coords(landmarks.landmark[373], w, h)
        return dist.euclidean(brow_pt, eye_pt)

    def _to_coords(self, lm, w, h):
        return (int(lm.x * w), int(lm.y * h))

    def detect_state(self, landmarks, w, h):
        coords = lambda indices: [self._to_coords(landmarks.landmark[i], w, h) for i in indices]

        # 1. Tính EAR (Mắt)
        left_eye = coords(LEFT_EYE)
        right_eye = coords(RIGHT_EYE)
        ear = (self._get_aspect_ratio(left_eye) + self._get_aspect_ratio(right_eye)) / 2.0

        # 2. Tính MAR (Miệng)
        mar = self._get_mouth_ratio(landmarks, w, h)

        #Logic suy luận cảm xúc
        #Kiểm tra Chớp mắt / Ngủ 
        if ear < self.BLINK_THRESH:
            return "BLINKING / EYES CLOSED", (0, 0, 255) # Đỏ

        #Kiểm tra Ngạc nhiên 
        # Thường ngạc nhiên MAR rất cao (> 0.6) và EAR bình thường/cao
        if mar > self.SURPRISE_THRESH:
            return "SURPRISED", (255, 255, 0) # Xanh lơ

        # Kiểm tra Cười
        # Để chính xác hơn cần đo độ cong khóe môi, nhưng MAR khoảng 0.3-0.5 thường là cười
        if 0.3 < mar <= self.SURPRISE_THRESH:
            return "SMILING", (0, 255, 0) # Xanh lá

        #Kiểm tra Tức giận
        # Đo khoảng cách giữa 2 đầu lông mày (điểm 107 và 336) hoặc khoảng cách mắt-lông mày giảm
        # Ở đây dùng logic đơn giản: Khóe môi đi xuống (Buồn/Khóc)
        # Xác định khóe môi (61, 291) so với trung bình môi (0, 17)
        p61 = landmarks.landmark[61].y
        p291 = landmarks.landmark[291].y
        p0 = landmarks.landmark[0].y # Môi trên
        p17 = landmarks.landmark[17].y # Môi dưới
        
        avg_lip_y = (p0 + p17) / 2
        # Nếu khóe môi thấp hơn đáng kể so với trung tâm môi -> Buồn/Mếu
        if p61 > avg_lip_y + 0.02 and p291 > avg_lip_y + 0.02: 
             return "SAD / CRYING", (255, 0, 255) # Tím

        return "NEUTRAL", (255, 255, 255) # Trắng
