import cv2
import os
import numpy as np
import albumentations as A

# --- CẤU HÌNH ---
INPUT_DIR = r"D:\Hoc Ki Cuoi\Capstone-project-VKU\DataAugmentation\input\sal"
OUTPUT_DIR =r"D:\Hoc Ki Cuoi\Capstone-project-VKU\DataAugmentation\output\Tangcuong_sal"
NUM_AUGMENTED_IMAGES_PER_ORIGINAL = 5  # Số ảnh mới tạo từ mỗi ảnh gốc

os.makedirs(OUTPUT_DIR, exist_ok=True)
# -------------------

# --- ĐỊNH NGHĨA PHÉP BIẾN ĐỔI (augmentation pipeline) ---
transform = A.Compose([
    # 1️⃣ Biến đổi hình học
    A.Rotate(limit=30, p=0.8),  # Xoay ngẫu nhiên ±30 độ
    A.ShiftScaleRotate(
        shift_limit=0.2, scale_limit=0.15, rotate_limit=0,
        p=0.8, border_mode=cv2.BORDER_REPLICATE  # Không để viền đen
    ),
    A.HorizontalFlip(p=0.5),  # Lật ngang 50%

    # 2️⃣ Biến đổi màu sắc
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7),

    # 3️⃣ Tăng cường bằng làm mờ / nhiễu
    A.GaussNoise(p=0.3),        # Nhiễu Gaussian nhẹ
    A.MotionBlur(blur_limit=3, p=0.2),  # Làm mờ nhẹ khi chuyển động


])

print("[INFO] Bắt đầu quá trình tăng cường dữ liệu...")

# --- XỬ LÝ TỪNG ẢNH ---
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_original_images = len(image_files)
processed_count = 0

for filename in image_files:
    image_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"[CẢNH BÁO] Không thể đọc ảnh: {image_path}. Bỏ qua.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(filename)[0]

    # Sinh ảnh mới
    for i in range(NUM_AUGMENTED_IMAGES_PER_ORIGINAL):
        augmented = transform(image=image)
        augmented_image = augmented['image']

        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        output_filename = os.path.join(OUTPUT_DIR, f"{base_name}_aug_{i+1}.jpg")
        cv2.imwrite(output_filename, augmented_image)

    processed_count += 1
    if processed_count % 10 == 0 or processed_count == total_original_images:
        print(f"[INFO] Đã xử lý {processed_count}/{total_original_images} ảnh gốc.")

# --- THỐNG KÊ ---
total_augmented_count = len(os.listdir(OUTPUT_DIR))
print(f"[INFO] Hoàn thành tăng cường dữ liệu.")
print(f"[INFO] Tổng số ảnh mới được tạo: {total_augmented_count}")
print(f"[INFO] Tổng (gốc + mới): {total_original_images + total_augmented_count}")

