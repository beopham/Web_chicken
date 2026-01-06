import os
from PIL import Image
import imagehash

folder = r"D:\Hoc Ki Cuoi\Web_chicken\Data_Phan_Ga_Not_Phan_Ga\Non-Poop"  # ✅ thư mục ảnh


hash_dict = {}

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    # Bỏ qua file không phải ảnh
    try:
        img = Image.open(path)
    except:
        continue

    # Tạo hash hình
    h = str(imagehash.average_hash(img))

    if h in hash_dict:
        hash_dict[h].append(file)
    else:
        hash_dict[h] = [file]

# In ra các nhóm ảnh trùng nhau
print("==== DUPLICATE IMAGES ====")
for h, files in hash_dict.items():
    if len(files) > 1:
        print("\n>> Hash:", h)
        for f in files:
            print("   -", f)
