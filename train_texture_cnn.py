import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import sys

# Tự động sửa đường dẫn import để tránh lỗi "ModuleNotFoundError"
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from models.texture_cnn import TextureCNN
except ImportError:
    # Fallback nếu cấu trúc thư mục khác biệt
    sys.path.append(os.path.dirname(current_dir))
    from models.texture_cnn import TextureCNN

def train():
    # 1. Cấu hình
    # Trỏ thẳng vào thư mục chứa code hiện tại + /data/train
    DATA_DIR = os.path.join(current_dir, 'data', 'train')
    MODEL_DIR = os.path.join(current_dir, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.pth')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 0.001

    print(f"[INFO] Đang tìm dữ liệu tại: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print("[LỖI] Không thấy thư mục data/train. Bạn đã chạy fix_structure.py chưa?")
        return

    # 2. Xử lý ảnh (QUAN TRỌNG: Resize về 64x64 để khớp với Model)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except Exception as e:
        print(f"[LỖI LOAD DATA] {e}")
        return

    print(f"[INFO] Tìm thấy {len(train_data)} ảnh. Nhãn: {train_data.class_to_idx}")
    
    if len(train_data.classes) != 2:
        print(f"[CẢNH BÁO] Số lớp dữ liệu là {len(train_data.classes)} (Cần 2: real/fake).")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Khởi tạo Model
    print(f"[INFO] Khởi tạo model trên {DEVICE}...")
    model = TextureCNN(num_classes=len(train_data.classes)).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    print("Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except RuntimeError as e:
                print(f"\n[LỖI KHI TRAIN] {e}")
                print("Khả năng cao là lỗi kích thước ảnh. Hãy chắc chắn transforms.Resize((64,64)).")
                return

        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # 5. Lưu model
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[THÀNH CÔNG] Đã lưu model: {MODEL_PATH}")

if __name__ == '__main__':
    train()