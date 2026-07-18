import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa mô hình CNN đơn giản để phân loại ảnh texture1
class TextureCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(TextureCNN, self).__init__()
        # Input: Ảnh màu (3 kênh RGB) kích thước 64x64
        
        # Layer 1: 64x64 -> 32x32 (sau pool)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 2: 32x32 -> 16x16 (sau pool)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Layer 3: 16x16 -> 8x8 (sau pool)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max Pooling (giảm kích thước đi 2 lần)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layer
        # Tính toán: 64 kênh * 8 * 8 (kích thước ảnh cuối cùng) = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # 128 neurons
        self.dropout = nn.Dropout(0.5) # Dropout để tránh overfitting
        self.fc2 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        # Qua 3 tầng Conv + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 16x16 -> 8x8
        
        # Duỗi ảnh ra (Flatten)
        x = x.view(-1, 64 * 8 * 8) 
        
        # Phân loại
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x