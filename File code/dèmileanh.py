import os

folder =r"D:\Hoc Ki Cuoi\Capstone-project-VKU\Data_new\Coccidiosis"

# Các đuôi file ảnh muốn tính
image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

count = 0

for file in os.listdir(folder):
    ext = os.path.splitext(file)[1].lower()
    if ext in image_exts:
        count += 1

print("✅ Số lượng ảnh trong thư mục:", count)
