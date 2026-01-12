from google import genai
from google.genai import types

# Khởi tạo Client với API Key của bạn
client = genai.Client(api_key="AIzaSyAQy7PuG06ScFSGoUKVoEBdXOagjvPxRuw")

# Danh sách các link "nguồn tri thức" của bạn
REFERENCE_LINKS = [
    "https://goovetvn.com/tin/cach-tri-benh-ga-ru-bang-toi.html",
    "https://khoathuy.vnua.edu.vn/4036.html"
]


def test_chat_with_url():
    print("--- CHƯƠNG TRÌNH TEST AI ĐỌC LINK CHUYÊN GIA ---")
    print("Đang kết nối với Gemini 2.0 Flash...")

    while True:
        # Nhập câu hỏi từ bàn phím
        user_input = input("\nBạn muốn hỏi gì về bệnh gà? (Gõ 'exit' để thoát): ")

        if user_input.lower() == 'exit':
            break

        try:
            # Gửi yêu cầu cho Gemini kèm theo các link tham khảo
            # Gemini sẽ tự động truy cập vào các URL này để lấy thông tin
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    f"Người dân hỏi: {user_input}. Hãy dùng thông tin từ các link sau để trả lời chính xác, ngắn gọn:",
                    *REFERENCE_LINKS  # Đưa danh sách link vào nội dung gửi đi
                ]
            )

            print("\nBÁC SĨ AI TRẢ LỜI:")
            print(response.text)

        except Exception as e:
            print(f"Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    test_chat_with_url()