import os
from google import genai

# Thiết lập API Key Tier 1 của bạn
os.environ["GOOGLE_API_KEY"] = ""

try:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    print("--- DANH SÁCH CÁC MODEL BẠN CÓ THỂ DÙNG ---")

    # Lấy danh sách model
    model_list = client.models.list()

    for model in model_list:
        # In ra tên và mô tả để kiểm tra
        print(f"Model ID: {model.name}")
        print(f"  - Tên hiển thị: {model.display_name}")
        # Kiểm tra xem có hỗ trợ tạo nội dung không
        if hasattr(model, 'supported_generation_methods'):
            print(f"  - Phương thức: {model.supported_generation_methods}")

        print("-" * 40)

except Exception as e:
    print(f"!!! Lỗi chi tiết: {e}")