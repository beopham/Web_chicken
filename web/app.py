# import os
# import io
# import re
# import base64
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import mysql.connector
# import os
#
#
# EMBEDDING_MODEL = "text-embedding-004"
# # 🔑 CẤU HÌNH API KEY (Thay bằng key thật của bạn)
# # LangChain/Google SDK sẽ đọc biến này
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDMV72De3esk0KpzpLLEo7PJRWTnwM8vr8"
# VECTOR_STORE = None # Biến toàn cục chứa cơ sở dữ liệu vector
# # --- CẦN CÀI ĐẶT: pip install flask tensorflow pillow numpy mysql-connector-python ---
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
# except ImportError:
#     print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
#
# # =================================================================
# # 1. CẤU HÌNH DATABASE
# # =================================================================
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '123456',
#     'database': 'benh_ga'
# }
#
#
# # --- HÀM KẾT NỐI DATABASE ---
# def get_db_connection():
#     """Thiết lập kết nối với MySQL"""
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         return conn
#     except mysql.connector.Error as err:
#         print(f"Lỗi kết nối database: {err}")
#         return None
#
#
# # =================================================================
# # 2. CẤU HÌNH VÀ TẢI MÔ HÌNH AI
# # =================================================================
#
# # ĐƯỜNG DẪN MÔ HÌNH CỦA BẠN
# MODEL_PATH = r'D:\Hoc Ki Cuoi\Capstone-project-VKU\Web_Final_ok\model\best_model.keras'
#
# # Tên 4 lớp bệnh (PHẢI ĐÚNG THỨ TỰ ABC)
# CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
# # Kích thước ảnh đầu vào
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
#
# app = Flask(__name__)
# # KHAI BÁO KEY BÍ MẬT CHO SESSION (Bắt buộc)
# app.secret_key = 'mot_chuoi_bi_mat_rat_dai_va_kho'
#
# # TẢI MÔ HÌNH (Chỉ 1 lần)
# model = None
# try:
#     if tf is not None:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         print(f">>> ✅ Mô hình AI đã được tải thành công từ: {MODEL_PATH}")
# except Exception as e:
#     print(f"!!! LỖI: Không thể tải mô hình từ đường dẫn: {MODEL_PATH}")
#     print(f"Lỗi chi tiết: {e}")
#     print("Chức năng chẩn đoán AI sẽ không hoạt động.")
#
#
# # -------------------------------------------------------------------------
# ## --- HÀM TIỀN XỬ LÝ ẢNH (DÙNG CHO CHẨN ĐOÁN AI) ---
# # -------------------------------------------------------------------------
# def process_and_predict(base64_img_string):
#     """Xử lý chuỗi base64 thành ảnh, chuẩn hóa và đưa ra dự đoán."""
#     if model is None:
#         return "Lỗi tải mô hình", 0.0
#
#     try:
#         # Xóa phần header base64 (ví dụ: data:image/jpeg;base64,)
#         img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
#         img_bytes = base64.b64decode(img_data)
#
#         # Chuyển bytes thành ảnh và resize
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
#
#         # Chuẩn bị cho mô hình
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#
#         # CHUẨN HÓA (Phải khớp với quá trình huấn luyện)
#         x = x / 255.0
#
#         # Dự đoán
#         predictions = model.predict(x)
#
#         # Lấy kết quả
#         predicted_class_index = np.argmax(predictions[0])
#         confidence = np.max(predictions[0]) * 100
#         predicted_class_name = CLASS_NAMES[predicted_class_index]
#
#         return predicted_class_name, confidence
#
#     except Exception as e:
#         print(f"LỖI xử lý ảnh/dự đoán: {e}")
#         return "Lỗi xử lý", 0.0
#
#
# # =================================================================
# # 3. ĐỊNH TUYẾN (ROUTES) ỨNG DỤNG
# # =================================================================
#
# ## --- ROUTE ĐĂNG NHẬP (Login) ---
# @app.route('/', methods=['GET', 'POST'])
# @app.route('/login', methods=['GET', 'POST'])
# def login_page():
#     error_message = None
#     if request.method == 'POST':
#         taikhoan = request.form.get('taikhoan')
#         mk = request.form.get('mk')
#
#         conn = get_db_connection()
#         if conn:
#             cursor = conn.cursor(dictionary=True)
#             query = "SELECT * FROM user WHERE taikhoan = %s AND matkhau = %s"
#             cursor.execute(query, (taikhoan, mk))
#             user = cursor.fetchone()
#
#             cursor.close()
#             conn.close()
#
#             if user:
#                 session['loggedin'] = True
#                 session['username'] = user.get('taikhoan')
#                 return redirect(url_for('trangchu_page'))
#             else:
#                 error_message = 'Tài khoản hoặc mật khẩu không đúng.'
#         else:
#             error_message = 'Lỗi kết nối cơ sở dữ liệu. Vui lòng kiểm tra lại cấu hình DB.'
#
#     # Hiển thị trang login
#     return render_template('login.html', error=error_message)
#
#
# ## --- ROUTE TRANG CHỦ (Sau khi đăng nhập) ---
# @app.route('/trangchu')
# def trangchu_page():
#     # Bảo vệ trang
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#
#     return render_template('trangchu.html', username=session.get('username'))
#
#
# ## --- ROUTE PHÂN LOẠI BỆNH GÀ (Form tải ảnh) ---
# @app.route('/phan_loai_benh_ga')
# def phan_loai_benh_ga_page():
#     # Bảo vệ trang
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#
#     # Trang này chứa form HTML/JS để gửi ảnh lên API /diagnose
#     return render_template('phan_loai_benh_ga.html', username=session.get('username'))
#
#
# ## --- ROUTE XỬ LÝ CHẨN ĐOÁN (API) ---
# @app.route('/diagnose', methods=['POST'])
# def diagnose():
#     """API nhận ảnh (base64) và trả về kết quả JSON."""
#     # 1. Bảo vệ API
#     if 'loggedin' not in session:
#         return jsonify({'error': 'Bạn cần đăng nhập để thực hiện chẩn đoán.'}), 401
#
#     # 2. Kiểm tra mô hình
#     if model is None:
#         return jsonify({'error': 'Mô hình AI chưa được tải thành công. Không thể chẩn đoán.'}), 500
#
#     try:
#         # Lấy dữ liệu ảnh Base64 từ yêu cầu POST (dạng JSON)
#         data = request.get_json()
#         img_data_b64 = data.get('image')
#
#         if not img_data_b64:
#             return jsonify({'error': 'Không tìm thấy dữ liệu ảnh (base64).'}), 400
#
#         # Thực hiện dự đoán
#         predicted_name, confidence = process_and_predict(img_data_b64)
#
#         if predicted_name in ["Lỗi xử lý", "Lỗi tải mô hình"]:
#             return jsonify({'error': f'Lỗi hệ thống trong quá trình dự đoán: {predicted_name}'}), 500
#
#         # Trả về kết quả JSON
#         return jsonify({
#             'success': True,
#             'prediction': {
#                 'disease': predicted_name,
#                 'confidence': f'{confidence:.2f}%'
#             }
#         })
#
#     except Exception as e:
#         print(f"LỖI SERVER trong route /diagnose: {e}")
#         return jsonify({'error': f'Lỗi server không xác định: {str(e)}'}), 500
#
#
# # =================================================================
# # 4. CHẠY ỨNG DỤNG
# # =================================================================
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




import os
import io
import re
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector

# =================================================================
# THAY ĐỔI LỚN: SỬ DỤNG CÁCH IMPORT LANGCHAIN HIỆN ĐẠI
# =================================================================
from google import genai
# Thay thế lỗi cũ bằng cách import mới và chính xác:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma # Dùng trực tiếp langchain-chroma thay vì langchain.vectorstores

# --- CẦN CÀI ĐẶT: đảm bảo đã cài đủ google-genai, langchain-google-genai, langchain-text-splitters, langchain-chroma ---
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
except ImportError:
    print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
    tf = None

# =================================================================
# 0. CẤU HÌNH VÀ KHỞI TẠO CHUNG
# =================================================================

os.environ["GOOGLE_API_KEY"] = "AIzaSyDMV72De3esk0KpzpLLEo7PJRWTnwM8vr8"
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

try:
    gemini_client = genai.Client()
except Exception as e:
    print(f"!!! LỖI KHỞI TẠO GEMINI CLIENT: {e}")

VECTOR_STORE = None
ACTIVE_CHATS = {}

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'benh_ga'
}

MODEL_PATH = r'D:\Hoc Ki Cuoi\Capstone-project-VKU\Web_Final_ok\model\best_model.keras'
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
IMG_HEIGHT = 224
IMG_WIDTH = 224

app = Flask(__name__)
app.secret_key = 'mot_chuoi_bi_mat_rat_dai_va_kho'

model = None
if tf is not None:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f">>> ✅ Mô hình AI đã được tải thành công từ: {MODEL_PATH}")
    except Exception as e:
        print(f"!!! LỖI: Không thể tải mô hình từ đường dẫn: {MODEL_PATH}")
        print(f"Lỗi chi tiết: {e}")
        print("Chức năng chẩn đoán AI sẽ không hoạt động.")


# =================================================================
# 1. HÀM DATABASE VÀ RAG
# =================================================================

def get_db_connection():
    """Thiết lập kết nối với MySQL"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Lỗi kết nối database: {err}")
        return None


def load_and_chunk_data():
    """Tải dữ liệu từ DB, chia chunks và tạo Vector Store (Chroma)."""
    global VECTOR_STORE

    if VECTOR_STORE is not None:
        print(">>> ✅ Vector Store đã được tải. Bỏ qua khởi tạo.")
        return

    conn = get_db_connection()
    if not conn:
        print("!!! LỖI: Không thể kết nối DB để tải dữ liệu RAG.")
        return

    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT ten_benh, dulieubenh FROM benh"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        texts = []
        for row in data:
            # Gộp tên bệnh vào nội dung để làm ngữ cảnh
            full_text = f"Tên bệnh: {row['ten_benh']}\n\nChi tiết: {row['dulieubenh']}"
            texts.append(full_text)

        # Chia Chunks (Sử dụng LangChain Text Splitters)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

        chunks = text_splitter.create_documents(texts)

        # Tạo Embeddings và Vector Store (Sử dụng langchain_chroma)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        VECTOR_STORE = Chroma.from_documents(chunks, embeddings)
        print(">>> ✅ Cơ sở dữ liệu Vector RAG (Chroma) đã được khởi tạo thành công.")

    except Exception as e:
        print(f"!!! LỖI TẠO RAG/VECTOR STORE: {e}")
        VECTOR_STORE = None


# --- CHẠY HÀM KHỞI TẠO RAG KHI ỨNG DỤNG KHỞI ĐỘNG ---
@app.before_request
def initialize_rag():
    """Chạy hàm khởi tạo RAG trước khi xử lý yêu cầu đầu tiên."""
    if VECTOR_STORE is None:
        load_and_chunk_data()


# =================================================================
# 2. HÀM XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# =================================================================

def process_and_predict(base64_img_string):
    """Xử lý chuỗi base64 thành ảnh, chuẩn hóa và đưa ra dự đoán."""
    if model is None:
        return "Lỗi tải mô hình", 0.0

    try:
        img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
        img_bytes = base64.b64decode(img_data)
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # CHUẨN HÓA

        predictions = model.predict(x)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return predicted_class_name, confidence

    except Exception as e:
        print(f"LỖI xử lý ảnh/dự đoán: {e}")
        return "Lỗi xử lý", 0.0


# =================================================================
# 3. ĐỊNH TUYẾN (ROUTES) ỨNG DỤNG
# =================================================================

# --- ROUTE ĐĂNG NHẬP (Login) --- (Giữ nguyên)
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error_message = None
    if request.method == 'POST':
        taikhoan = request.form.get('taikhoan')
        mk = request.form.get('mk')

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s"
            cursor.execute(query, (taikhoan, mk))
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            if user:
                session['loggedin'] = True
                session['user_id'] = user.get('idTaikhoan')
                session['username'] = user.get('taikhoan')
                return redirect(url_for('trangchu_page'))
            else:
                error_message = 'Tài khoản hoặc mật khẩu không đúng.'
        else:
            error_message = 'Lỗi kết nối cơ sở dữ liệu. Vui lòng kiểm tra lại cấu hình DB.'

    return render_template('login.html', error=error_message)


# --- ROUTE TRANG CHỦ & PHÂN LOẠI (Giữ nguyên) ---
@app.route('/trangchu')
def trangchu_page():
    if 'loggedin' not in session:
        return redirect(url_for('login_page'))
    return render_template('trangchu.html', username=session.get('username'))


@app.route('/phan_loai_benh_ga')
def phan_loai_benh_ga_page():
    if 'loggedin' not in session:
        return redirect(url_for('login_page'))
    return render_template('phan_loai_benh_ga.html', username=session.get('username'))


# --- ROUTE XỬ LÝ CHẨN ĐOÁN VÀ KHỞI TẠO CHAT (TÍCH HỢP RAG) ---
@app.route('/diagnose', methods=['POST'])
def diagnose_and_start_chat():
    """Chẩn đoán ảnh, khởi tạo chat session, và trả về phản hồi RAG đầu tiên."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Bạn cần đăng nhập để thực hiện chẩn đoán.'}), 401

    if VECTOR_STORE is None:
        return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500

    try:
        # 1. Chẩn đoán ảnh
        data = request.get_json()
        img_data_b64 = data.get('image')
        if not img_data_b64:
            return jsonify({'error': 'Không tìm thấy dữ liệu ảnh (base64).'}), 400

        predicted_name, confidence = process_and_predict(img_data_b64)

        # Xử lý trường hợp Healthy hoặc Lỗi dự đoán
        if predicted_name in ["Lỗi xử lý", "Lỗi tải mô hình"]:
            return jsonify({
                'success': True,
                'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
                'initial_chat_response': f"Lỗi hệ thống khi dự đoán."
            })

        if predicted_name == "Healthy":
            return jsonify({
                'success': True,
                'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
                'initial_chat_response': f"Kết quả chẩn đoán: **{predicted_name}**. Gà của bạn khỏe mạnh, hãy tiếp tục duy trì chế độ chăm sóc tốt!"
            })

        # 2. Khởi tạo Chatbot và RAG cho bệnh được chẩn đoán

        # a. Truy vấn RAG (Lấy thông tin phác đồ ban đầu)
        query_rag = f"Phác đồ điều trị ban đầu và triệu chứng chính của bệnh {predicted_name}"
        rag_docs = VECTOR_STORE.similarity_search(query_rag, k=3)
        rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])

        # b. Thiết lập System Prompt và Message
        system_prompt = (
            "Bạn là chuyên gia thú y gia cầm. Hãy cung cấp tư vấn chi tiết DỰA TRÊN NGỮ CẢNH "
            "CHUYÊN MÔN được cung cấp. Tuyệt đối không tự suy diễn nếu không có dữ liệu. "
            "Sử dụng Markdown để định dạng câu trả lời dễ đọc."
        )
        initial_prompt = (
            f"Kết quả chẩn đoán hình ảnh là **{predicted_name}** với độ tin cậy {confidence:.2f}%. "
            f"Dưới đây là các dữ liệu chuyên môn về bệnh này:\n\n"
            f"--- NGUỒN DỮ LIỆU RAG ---\n{rag_context}\n--- END RAG ---\n\n"
            "Dựa vào thông tin trên, hãy đưa ra: 1. Tóm tắt nhanh về bệnh. 2. Phác đồ điều trị khẩn cấp ban đầu (thuốc và cách ly)."
        )

        # c. Khởi tạo Chat Session với System Prompt
        chat = gemini_client.chats.create(
            model=LLM_MODEL,
            config={'system_instruction': system_prompt}
        )

        # d. Gửi tin nhắn đầu tiên và lưu Session
        initial_response = chat.send_message(initial_prompt)
        ACTIVE_CHATS[user_id] = chat  # Lưu lại phiên chat

        # 3. Trả về kết quả JSON và Phản hồi Chatbot
        return jsonify({
            'success': True,
            'prediction': {
                'disease': predicted_name,
                'confidence': f'{confidence:.2f}%'
            },
            'initial_chat_response': initial_response.text
        })

    except Exception as e:
        print(f"LỖI KHỞI TẠO CHAT VÀ RAG: {e}")
        # Xóa chat session nếu lỗi
        if user_id in ACTIVE_CHATS: del ACTIVE_CHATS[user_id]
        return jsonify({'error': f'Lỗi hệ thống khi khởi tạo tư vấn: {str(e)}'}), 500


# --- ROUTE XỬ LÝ CÂU HỎI TIẾP THEO ---
@app.route('/chat', methods=['POST'])
def handle_followup_chat():
    """API nhận câu hỏi tiếp theo, sử dụng RAG và phiên chat đã lưu."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Bạn cần đăng nhập.'}), 401

    if user_id not in ACTIVE_CHATS:
        return jsonify({'error': 'Chưa có phiên chat nào được khởi tạo. Vui lòng chẩn đoán trước.'}), 400

    if VECTOR_STORE is None:
        return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500

    try:
        data = request.get_json()
        user_question = data.get('question')

        if not user_question:
            return jsonify({'error': 'Không tìm thấy câu hỏi.'}), 400

        # Lấy phiên chat hiện tại
        current_chat = ACTIVE_CHATS[user_id]

        # 1. Truy vấn RAG (Lấy thông tin liên quan đến câu hỏi mới)
        rag_docs = VECTOR_STORE.similarity_search(user_question, k=3)
        rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])

        # 2. Gộp RAG vào Prompt
        augmented_prompt = (
            f"Dựa trên NGỮ CẢNH BỔ SUNG dưới đây, hãy trả lời câu hỏi: '{user_question}'. "
            f"Đảm bảo câu trả lời nhất quán với lịch sử trò chuyện (nếu có).\n\n"
            f"--- NGỮ CẢNH RAG ---\n{rag_context}\n--- END NGỮ CẢNH ---\n"
        )

        # 3. Gửi Prompt Tăng cường đến Gemini (giữ lại lịch sử chat)
        response = current_chat.send_message(augmented_prompt)

        # 4. Trả về kết quả JSON
        return jsonify({
            'success': True,
            'response': response.text
        })

    except Exception as e:
        print(f"LỖI XỬ LÝ CHAT TIẾP THEO: {e}")
        return jsonify({'error': f'Lỗi server không xác định: {str(e)}'}), 500


# =================================================================
# 4. CHẠY ỨNG DỤNG
# =================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)