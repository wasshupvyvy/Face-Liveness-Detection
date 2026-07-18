import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from models.texture_cnn import TextureCNN
import os

# TextureDetector sử dụng mô hình CNN để phân loại ảnh mặt người là thật hay giả
class TextureDetector:
    # Khởi tạo bộ phát hiện với đường dẫn mô hình đã huấn luyện
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = TextureCNN(num_classes=2).to(self.device) # Khởi tạo mô hình
        self.loaded = False
        # Tải trọng số mô hình đã huấn luyện
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)) # Load model
            self.model.eval() # Chuyển sang chế độ đánh giá
            self.loaded = True 
            print(f"[INFO] Model loaded: {model_path}")
        else:
            print(f"[WARNING] Model not found: {model_path}. Please train first!")

        # Thiết lập biến đổi ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize chuẩn input của model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Dự đoán độ thật của ảnh mặt (trả về xác suất là thật từ 0.0 đến 1.0)
    def predict(self, face_bgr):
        if not self.loaded: return 0.0 # Nếu model chưa load, trả về 0.0
        
        # Chuyển đổi sang format model hiểu
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) #   Chuyển BGR sang RGB
        pil_img = Image.fromarray(img_rgb) # Chuyển sang PIL Image
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device) # Thêm batch dimension và chuyển device
        
        # Dự đoán với mô hình
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            real_score = probs[0][1].item() # Giả sử index 1 là Real
            
        return real_score