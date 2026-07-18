import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Lấy đường dẫn của file evaluate.py đang chạy
current_dir = os.path.dirname(os.path.abspath(__file__))

# Thêm đường dẫn này vào danh sách tìm kiếm của Python
# Giúp Python nhìn thấy folder 'models' nằm cùng chỗ
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Bây giờ mới import model
try:
    from models.texture_cnn import TextureCNN
except ImportError:
    # Thử fallback: Nếu file này nằm trong folder con (ví dụ utils), nhảy ra ngoài 1 cấp
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from models.texture_cnn import TextureCNN

# Hàm đánh giá mô hình trên tập test
def evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Sử dụng GPU nếu có
    print(f"[INFO] Đang chạy trên thiết bị: {DEVICE}") # In thiết bị
    
    # Đường dẫn đến folder test và file model
    test_dir = os.path.join(current_dir, 'data', 'test')
    model_path = os.path.join(current_dir, 'models', 'trained_model.pth')

    # In thông tin
    print(f"[INFO] Dữ liệu test: {test_dir}")
    print(f"[INFO] Model load từ: {model_path}")

    # Kiểm tra tồn tại
    if not os.path.exists(test_dir):
        print(f"[LỖI]Không tìm thấy thư mục test: {test_dir}")
        print("👉 Bạn hãy chạy file 'split_train_test_v2.py' để tạo dữ liệu test trước đã!")
        return

    # Kiểm tra file model
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file model: {model_path}")
        print("Bạn cần chạy file 'train_texture_cnn.py' để huấn luyện xong mới có model.")
        return

    #LOAD DỮ LIỆU
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Chuẩn hóa
    ])

    try: # Đọc dữ liệu test
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)  # Load dữ liệu
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"[LỖI] Không đọc được dữ liệu ảnh: {e}")
        return

    # In ra các lớp tìm thấy
    print(f"[INFO] Các nhãn trong tập test: {test_data.class_to_idx}")
    # Lưu ý: Số lượng class (num_classes) phải khớp với lúc train
    num_classes = len(test_data.classes)

    #LOAD MODEL & ĐÁNH GIÁ
    try:
        model = TextureCNN(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Chuyển sang chế độ đánh giá (quan trọng!)
    except Exception as e:
        print(f"[LỖI LOAD MODEL] Có thể số lượng lớp (num_classes) không khớp hoặc file lỗi.")
        print(f"Chi tiết: {e}")
        return

    correct = 0
    total = 0
    
    print("\nĐang chấm điểm...") 
    
    with torch.no_grad(): # Không tính toán đạo hàm cho nhẹ
        for images, labels in test_loader: # Lặp qua từng batch
            images, labels = images.to(DEVICE), labels.to(DEVICE) # Đưa lên GPU nếu có
            
            outputs = model(images) # Dự đoán
            _, predicted = torch.max(outputs.data, 1) # Lấy nhãn dự đoán
            
            total += labels.size(0) # Cộng số ảnh đã đo
            correct += (predicted == labels).sum().item() # Cộng số dự đoán đúng

    if total > 0: # Tránh chia cho 0
        accuracy = 100 * correct / total # Tính độ chính xác
        print("="*50)
        print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN {total} ẢNH:") # In kết quả
        print(f"Độ chính xác (Accuracy): {accuracy:.2f}%") 
        print("="*50)
    else:
        print("[CẢNH BÁO] Không có ảnh nào trong tập test để chấm điểm.")

if __name__ == '__main__':
    evaluate()