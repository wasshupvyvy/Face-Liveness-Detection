from scipy.spatial import distance as dist

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Tính khoảng cách giữa các điểm chiều dọc mắt
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Tính khoảng cách chiều ngang mắt
    C = dist.euclidean(eye[0], eye[3])
    # Công thức EAR
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Tính độ mở của miệng (dựa trên các điểm môi trong)
    A = dist.euclidean(mouth[13], mouth[19]) # Chiều dọc
    B = dist.euclidean(mouth[14], mouth[18]) # Chiều dọc
    C = dist.euclidean(mouth[15], mouth[17]) # Chiều dọc
    D = dist.euclidean(mouth[12], mouth[16]) # Chiều ngang
    
    mar = (A + B + C) / (2.0 * D)
    return mar
