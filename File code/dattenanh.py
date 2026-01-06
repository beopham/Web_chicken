import os

# 🗂️ Đường dẫn tới thư mục chứa ảnh
folder =r"D:\Hoc Ki Cuoi\Web_chicken\Data_Phan_Ga_Not_Phan_Ga\Poop"  # ✅ thư mục ảnh

prefix = "Poop"  # tiền tố tên ảnh (mày có thể đổi tuỳ ý, ví - dụ: ncd, nc, newcastle...)

# Lấy danh sách tất cả ảnh trong thư mục (lọc đuôi ảnh)
files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
files.sort()  # sắp xếp để đặt tên theo thứ tự cố định

# Đổi tên từng ảnh
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    new_name = f"{prefix}{i:02d}.jpg"  # ncd01.jpg, ncd02.jpg,...
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)

print("✅ Đổi tên ảnh trong thư mục xong!")
