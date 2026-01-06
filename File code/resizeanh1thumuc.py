import os
from PIL import Image

# ✅ Thư mục chứa ảnh
folder = r"D:\Hoc Ki Cuoi\Web_chicken\Data_Phan_Ga_Not_Phan_Ga\Non-Poop"  # ✅ thư mục ảnh
# ✅ Kích thước mới
new_size = (224, 224)

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    # bỏ file không phải ảnh
    try:
        img = Image.open(path)
    except:
        print("❌ Bỏ qua (không phải ảnh):", file)
        continue

    # resize
    img = img.resize(new_size, Image.BILINEAR)
    img.save(path)   # ✅ ghi đè

    print("✅ Done:", file)

print("\n✅ Hoàn tất!")
