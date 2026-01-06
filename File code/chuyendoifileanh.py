import os
from PIL import Image

# --- BẠN CẦN THAY ĐỔI 2 ĐƯỜNG DẪN NÀY ---
source_folder = r"D:\Hoc Ki Cuoi\Capstone-project-VKU\Data_new\anh" # Đặt đường dẫn đến thư mục chứa ảnh gốc
output_folder = r"D:\Hoc Ki Cuoi\Capstone-project-VKU\Data_new\duoianhmoi"  # Đặt đường dẫn đến thư mục bạn muốn lưu ảnh JPG
# ----------------------------------------

# Các định dạng ảnh mà code sẽ tìm để chuyển đổi
# Bạn có thể thêm các định dạng khác nếu cần (ví dụ: '.gif', '.tiff')
VALID_IMAGE_FORMATS = ('.png', '.bmp', '.webp', '.jpeg', '.jfif', '.tif', '.tiff')


def convert_all_to_jpg(src_dir, out_dir, quality=90):
    """
    Chuyển đổi tất cả ảnh trong thư mục src_dir sang JPG và lưu vào out_dir.
    """
    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Đã tạo thư mục: {out_dir}')

    # Đếm số lượng file đã chuyển đổi
    converted_count = 0
    skipped_count = 0

    # Lặp qua tất cả các file trong thư mục nguồn
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)

        # Bỏ qua nếu là thư mục con
        if not os.path.isfile(file_path):
            continue

        # Lấy tên file và đuôi file (phần mở rộng)
        file_name_no_ext, file_ext = os.path.splitext(filename)

        # Kiểm tra xem có phải định dạng ảnh chúng ta muốn chuyển không
        if file_ext.lower() in VALID_IMAGE_FORMATS:

            # Nếu file đã là .jpg rồi thì bỏ qua
            if file_ext.lower() == '.jpg' or file_ext.lower() == '.jpeg':
                print(f'Bỏ qua (đã là JPG): {filename}')
                skipped_count += 1
                continue

            try:
                # Mở file ảnh
                with Image.open(file_path) as img:
                    # Tạo tên file mới
                    output_filename = file_name_no_ext + '.jpg'
                    output_path = os.path.join(out_dir, output_filename)

                    # Xử lý kênh màu (ví dụ: ảnh PNG có kênh Alpha 'RGBA')
                    # JPG không hỗ trợ trong suốt, nên cần chuyển về 'RGB'
                    if img.mode == 'RGBA' or img.mode == 'P':  # 'P' là chế độ palette
                        img = img.convert('RGB')

                    # Lưu file sang định dạng JPG với chất lượng 90%
                    img.save(output_path, 'JPEG', quality=quality)
                    print(f'Đã chuyển đổi: {filename} -> {output_filename}')
                    converted_count += 1

            except Exception as e:
                print(f'LỖI khi chuyển đổi {filename}: {e}')

        else:
            print(f'Bỏ qua (không phải định dạng ảnh): {filename}')

    print('\n--- HOÀN TẤT ---')
    print(f'Tổng số file đã chuyển đổi: {converted_count}')
    print(f'Tổng số file đã bỏ qua: {skipped_count}')


# --- Chạy hàm ---
if __name__ == "__main__":
    convert_all_to_jpg(source_folder, output_folder)