from PIL import Image
import os

# Thư mục chứa ảnh gốc
input_dir = r"D:\Hoc Ki Cuoi\Capstone-project-VKU\Data_new\New Castle Disease"

# Thư mục lưu ảnh sau khi resize
output_dir = r"D:\Hoc Ki Cuoi\Capstone-project-VKU\Data_new\New Castle Disease_new"

# Kích thước mới
new_size = (224, 224)

# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Duyệt file
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)

    # Chỉ xử lý file ảnh
    try:
        with Image.open(input_path) as img:
            img = img.resize(new_size, Image.BILINEAR)

            output_path = os.path.join(output_dir, file_name)
            img.save(output_path)

            print("✅ Done:", file_name)
    except:
        print("❌ Skip (not image):", file_name)

print("✅ Completed!")
