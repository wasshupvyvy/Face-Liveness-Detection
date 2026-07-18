
import os
import shutil
import random

def split_data_recursive(test_ratio=0.2):
    # 1. Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'data', 'train')
    test_dir = os.path.join(base_dir, 'data', 'test')

    print("="*60)
    print(f"CHIA DỮ LIỆU TỰ ĐỘNG (BẢN VÉT CẠN - RECURSIVE)")
    print(f"Nguồn: {train_dir}")
    print(f"Đích:  {test_dir}")
    print("="*60)

    if not os.path.exists(train_dir):
        print("LỖI: Không tìm thấy thư mục 'data/train'.")
        return

    # Lấy danh sách các lớp (real, fake, spoof...)
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    if not classes:
        print("LỖI: Thư mục 'data/train' rỗng! Bạn hãy copy ảnh vào đó trước.")
        return

    total_moved = 0

    for class_name in classes:
        src_class_path = os.path.join(train_dir, class_name)
        dst_class_path = os.path.join(test_dir, class_name)
        
        # --- KỸ THUẬT VÉT CẠN (Deep Scan) ---
        # Tìm tất cả file ảnh trong mọi thư mục con
        all_images = []
        for root, dirs, files in os.walk(src_class_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    full_path = os.path.join(root, file)
                    all_images.append(full_path)
        
        print(f"Đang quét lớp '{class_name}': Tìm thấy {len(all_images)} ảnh.")
        
        if len(all_images) == 0:
            print(f"CẢNH BÁO: Không có ảnh nào trong '{class_name}'. Bỏ qua.")
            continue

        # Tính toán số lượng cần chuyển
        num_to_move = int(len(all_images) * test_ratio)
        if num_to_move == 0:
            print("Quá ít ảnh để chia. Giữ nguyên.")
            continue

        # Xáo trộn và cắt
        random.shuffle(all_images)
        images_to_move = all_images[:num_to_move]
        
        print(f"Đang chuyển {num_to_move} ảnh sang Test...")
        
        # Tạo thư mục đích
        os.makedirs(dst_class_path, exist_ok=True)
        
        count = 0
        for src_file in images_to_move:
            try:
                filename = os.path.basename(src_file)
                # Xử lý trùng tên
                if os.path.exists(os.path.join(dst_class_path, filename)):
                    filename = f"moved_{random.randint(1000,9999)}_{filename}"
                
                dst_file = os.path.join(dst_class_path, filename)
                
                shutil.move(src_file, dst_file)
                count += 1
            except Exception as e:
                print(f"Lỗi: {e}")
        
        total_moved += count
        print(f"Đã chuyển xong {count} ảnh.")

    print("="*60)
    print(f"TỔNG KẾT: Đã chuyển {total_moved} ảnh sang 'data/test'.")
    if total_moved == 0:
        print("VẪN 0 ẢNH? -> Hãy kiểm tra lại folder 'data/train', chắc chắn nó đang rỗng!")
    print("="*60)

if __name__ == "__main__":
    split_data_recursive()