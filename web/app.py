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


#
# import os
# import io
# import re
# import base64
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import mysql.connector
#
# # =================================================================
# # THAY ĐỔI LỚN: SỬ DỤNG CÁCH IMPORT LANGCHAIN HIỆN ĐẠI
# # =================================================================
# from google import genai
# # Thay thế lỗi cũ bằng cách import mới và chính xác:
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma # Dùng trực tiếp langchain-chroma thay vì langchain.vectorstores
#
# # --- CẦN CÀI ĐẶT: đảm bảo đã cài đủ google-genai, langchain-google-genai, langchain-text-splitters, langchain-chroma ---
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
# except ImportError:
#     print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
#     tf = None
#
# # =================================================================
# # 0. CẤU HÌNH VÀ KHỞI TẠO CHUNG
# # =================================================================
#
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCifSb7b1ldIDPiSn7Gz2ZCmTm6HtaLbr0"
# EMBEDDING_MODEL = "text-embedding-004"
# LLM_MODEL = "gemini-2.5-flash"
#
# try:
#     gemini_client = genai.Client()
# except Exception as e:
#     print(f"!!! LỖI KHỞI TẠO GEMINI CLIENT: {e}")
#
# VECTOR_STORE = None
# ACTIVE_CHATS = {}
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '123456',
#     'database': 'benh_ga'
# }
#
# MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
# CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
#
# app = Flask(__name__)
# app.secret_key = 'mot_chuoi_bi_mat_rat_dai_va_kho'
#
# model = None
# if tf is not None:
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         print(f">>> ✅ Mô hình AI đã được tải thành công từ: {MODEL_PATH}")
#     except Exception as e:
#         print(f"!!! LỖI: Không thể tải mô hình từ đường dẫn: {MODEL_PATH}")
#         print(f"Lỗi chi tiết: {e}")
#         print("Chức năng chẩn đoán AI sẽ không hoạt động.")
#
#
# # =================================================================
# # 1. HÀM DATABASE VÀ RAG
# # =================================================================
#
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
# def load_and_chunk_data():
#     """Tải dữ liệu từ DB, chia chunks và tạo Vector Store (Chroma)."""
#     global VECTOR_STORE
#
#     if VECTOR_STORE is not None:
#         print(">>> ✅ Vector Store đã được tải. Bỏ qua khởi tạo.")
#         return
#
#     conn = get_db_connection()
#     if not conn:
#         print("!!! LỖI: Không thể kết nối DB để tải dữ liệu RAG.")
#         return
#
#     try:
#         cursor = conn.cursor(dictionary=True)
#         query = "SELECT ten_benh, dulieubenh FROM benh"
#         cursor.execute(query)
#         data = cursor.fetchall()
#         cursor.close()
#         conn.close()
#
#         texts = []
#         for row in data:
#             # Gộp tên bệnh vào nội dung để làm ngữ cảnh
#             full_text = f"Tên bệnh: {row['ten_benh']}\n\nChi tiết: {row['dulieubenh']}"
#             texts.append(full_text)
#
#         # Chia Chunks (Sử dụng LangChain Text Splitters)
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", ".", "!", "?"]
#         )
#
#         chunks = text_splitter.create_documents(texts)
#
#         # Tạo Embeddings và Vector Store (Sử dụng langchain_chroma)
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         VECTOR_STORE = Chroma.from_documents(chunks, embeddings)
#         print(">>> ✅ Cơ sở dữ liệu Vector RAG (Chroma) đã được khởi tạo thành công.")
#
#     except Exception as e:
#         print(f"!!! LỖI TẠO RAG/VECTOR STORE: {e}")
#         VECTOR_STORE = None
#
#
# # --- CHẠY HÀM KHỞI TẠO RAG KHI ỨNG DỤNG KHỞI ĐỘNG ---
# @app.before_request
# def initialize_rag():
#     """Chạy hàm khởi tạo RAG trước khi xử lý yêu cầu đầu tiên."""
#     if VECTOR_STORE is None:
#         load_and_chunk_data()
#
#
# # =================================================================
# # 2. HÀM XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# # =================================================================
#
# def process_and_predict(base64_img_string):
#     """Xử lý chuỗi base64 thành ảnh, chuẩn hóa và đưa ra dự đoán."""
#     if model is None:
#         return "Lỗi tải mô hình", 0.0
#
#     try:
#         img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
#         img_bytes = base64.b64decode(img_data)
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = x / 255.0  # CHUẨN HÓA
#
#         predictions = model.predict(x)
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
# # --- ROUTE ĐĂNG NHẬP (Login) --- (Giữ nguyên)
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
#             query = "SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s"
#             cursor.execute(query, (taikhoan, mk))
#             user = cursor.fetchone()
#
#             cursor.close()
#             conn.close()
#
#             if user:
#                 session['loggedin'] = True
#                 session['user_id'] = user.get('idTaikhoan')
#                 session['username'] = user.get('taikhoan')
#                 return redirect(url_for('trangchu_page'))
#             else:
#                 error_message = 'Tài khoản hoặc mật khẩu không đúng.'
#         else:
#             error_message = 'Lỗi kết nối cơ sở dữ liệu. Vui lòng kiểm tra lại cấu hình DB.'
#
#     return render_template('login.html', error=error_message)
#
#
# # --- ROUTE TRANG CHỦ & PHÂN LOẠI (Giữ nguyên) ---
# @app.route('/trangchu')
# def trangchu_page():
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('trangchu.html', username=session.get('username'))
#
#
# @app.route('/phan_loai_benh_ga')
# def phan_loai_benh_ga_page():
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('phan_loai_benh_ga.html', username=session.get('username'))
#
#
# # --- ROUTE XỬ LÝ CHẨN ĐOÁN VÀ KHỞI TẠO CHAT (TÍCH HỢP RAG) ---
# @app.route('/diagnose', methods=['POST'])
# def diagnose_and_start_chat():
#     """Chẩn đoán ảnh, khởi tạo chat session, và trả về phản hồi RAG đầu tiên."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Bạn cần đăng nhập để thực hiện chẩn đoán.'}), 401
#
#     if VECTOR_STORE is None:
#         return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500
#
#     try:
#         # 1. Chẩn đoán ảnh
#         data = request.get_json()
#         img_data_b64 = data.get('image')
#         if not img_data_b64:
#             return jsonify({'error': 'Không tìm thấy dữ liệu ảnh (base64).'}), 400
#
#         predicted_name, confidence = process_and_predict(img_data_b64)
#
#         # Xử lý trường hợp Healthy hoặc Lỗi dự đoán
#         if predicted_name in ["Lỗi xử lý", "Lỗi tải mô hình"]:
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': f"Lỗi hệ thống khi dự đoán."
#             })
#
#         if predicted_name == "Healthy":
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': f"Kết quả chẩn đoán: **{predicted_name}**. Gà của bạn khỏe mạnh, hãy tiếp tục duy trì chế độ chăm sóc tốt!"
#             })
#
#         # 2. Khởi tạo Chatbot và RAG cho bệnh được chẩn đoán
#
#         # a. Truy vấn RAG (Lấy thông tin phác đồ ban đầu)
#         query_rag = f"Phác đồ điều trị ban đầu và triệu chứng chính của bệnh {predicted_name}"
#         rag_docs = VECTOR_STORE.similarity_search(query_rag, k=3)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # b. Thiết lập System Prompt và Message
#         system_prompt = (
#             "Bạn là chuyên gia thú y gia cầm. Hãy cung cấp tư vấn chi tiết DỰA TRÊN NGỮ CẢNH "
#             "CHUYÊN MÔN được cung cấp. Tuyệt đối không tự suy diễn nếu không có dữ liệu. "
#             "Sử dụng Markdown để định dạng câu trả lời dễ đọc."
#         )
#         initial_prompt = (
#             f"Kết quả chẩn đoán hình ảnh là **{predicted_name}** với độ tin cậy {confidence:.2f}%. "
#             f"Dưới đây là các dữ liệu chuyên môn về bệnh này:\n\n"
#             f"--- NGUỒN DỮ LIỆU RAG ---\n{rag_context}\n--- END RAG ---\n\n"
#             "Dựa vào thông tin trên, hãy đưa ra: 1. Tóm tắt nhanh về bệnh. 2. Phác đồ điều trị khẩn cấp ban đầu (thuốc và cách ly)."
#         )
#
#         # c. Khởi tạo Chat Session với System Prompt
#         chat = gemini_client.chats.create(
#             model=LLM_MODEL,
#             config={'system_instruction': system_prompt}
#         )
#
#         # d. Gửi tin nhắn đầu tiên và lưu Session
#         initial_response = chat.send_message(initial_prompt)
#         ACTIVE_CHATS[user_id] = chat  # Lưu lại phiên chat
#
#         # 3. Trả về kết quả JSON và Phản hồi Chatbot
#         return jsonify({
#             'success': True,
#             'prediction': {
#                 'disease': predicted_name,
#                 'confidence': f'{confidence:.2f}%'
#             },
#             'initial_chat_response': initial_response.text
#         })
#
#     except Exception as e:
#         print(f"LỖI KHỞI TẠO CHAT VÀ RAG: {e}")
#         # Xóa chat session nếu lỗi
#         if user_id in ACTIVE_CHATS: del ACTIVE_CHATS[user_id]
#         return jsonify({'error': f'Lỗi hệ thống khi khởi tạo tư vấn: {str(e)}'}), 500
#
#
# # --- ROUTE XỬ LÝ CÂU HỎI TIẾP THEO ---
# @app.route('/chat', methods=['POST'])
# def handle_followup_chat():
#     """API nhận câu hỏi tiếp theo, sử dụng RAG và phiên chat đã lưu."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Bạn cần đăng nhập.'}), 401
#
#     if user_id not in ACTIVE_CHATS:
#         return jsonify({'error': 'Chưa có phiên chat nào được khởi tạo. Vui lòng chẩn đoán trước.'}), 400
#
#     if VECTOR_STORE is None:
#         return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500
#
#     try:
#         data = request.get_json()
#         user_question = data.get('question')
#
#         if not user_question:
#             return jsonify({'error': 'Không tìm thấy câu hỏi.'}), 400
#
#         # Lấy phiên chat hiện tại
#         current_chat = ACTIVE_CHATS[user_id]
#
#         # 1. Truy vấn RAG (Lấy thông tin liên quan đến câu hỏi mới)
#         rag_docs = VECTOR_STORE.similarity_search(user_question, k=3)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # 2. Gộp RAG vào Prompt
#         augmented_prompt = (
#             f"Dựa trên NGỮ CẢNH BỔ SUNG dưới đây, hãy trả lời câu hỏi: '{user_question}'. "
#             f"Đảm bảo câu trả lời nhất quán với lịch sử trò chuyện (nếu có).\n\n"
#             f"--- NGỮ CẢNH RAG ---\n{rag_context}\n--- END NGỮ CẢNH ---\n"
#         )
#
#         # 3. Gửi Prompt Tăng cường đến Gemini (giữ lại lịch sử chat)
#         response = current_chat.send_message(augmented_prompt)
#
#         # 4. Trả về kết quả JSON
#         return jsonify({
#             'success': True,
#             'response': response.text
#         })
#
#     except Exception as e:
#         print(f"LỖI XỬ LÝ CHAT TIẾP THEO: {e}")
#         return jsonify({'error': f'Lỗi server không xác định: {str(e)}'}), 500
#
#
# # =================================================================
# # 4. CHẠY ỨNG DỤNG
# # =================================================================
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
#
#
#
#
# import os
# import io
# import re
# import base64
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import mysql.connector
#
# # =================================================================
# # THAY ĐỔI LỚN: SỬ DỤNG CÁCH IMPORT LANGCHAIN HIỆN ĐẠI
# # =================================================================
# from google import genai
# # Thay thế lỗi cũ bằng cách import mới và chính xác:
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma # Dùng trực tiếp langchain-chroma thay vì langchain.vectorstores
#
# # --- CẦN CÀI ĐẶT: đảm bảo đã cài đủ google-genai, langchain-google-genai, langchain-text-splitters, langchain-chroma ---
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
# except ImportError:
#     print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
#     tf = None
#
# # =================================================================
# # 0. CẤU HÌNH VÀ KHỞI TẠO CHUNG
# # =================================================================
#
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCifSb7b1ldIDPiSn7Gz2ZCmTm6HtaLbr0"
# EMBEDDING_MODEL = "text-embedding-004"
# LLM_MODEL = "gemini-flash-latest"
# try:
#     gemini_client = genai.Client()
# except Exception as e:
#     print(f"!!! LỖI KHỞI TẠO GEMINI CLIENT: {e}")
#
# VECTOR_STORE = None
# ACTIVE_CHATS = {}
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '123456',
#     'database': 'benh_ga'
# }
#
# MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
# CLASS_NAMES = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
#
# app = Flask(__name__)
# app.secret_key = 'mot_chuoi_bi_mat_rat_dai_va_kho'
#
# model = None
# if tf is not None:
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         print(f">>> ✅ Mô hình AI đã được tải thành công từ: {MODEL_PATH}")
#     except Exception as e:
#         print(f"!!! LỖI: Không thể tải mô hình từ đường dẫn: {MODEL_PATH}")
#         print(f"Lỗi chi tiết: {e}")
#         print("Chức năng chẩn đoán AI sẽ không hoạt động.")
#
#
# # =================================================================
# # 1. HÀM DATABASE VÀ RAG
# # =================================================================
#
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
# def load_and_chunk_data():
#     """Tải dữ liệu từ DB, chia chunks và tạo Vector Store (Chroma)."""
#     global VECTOR_STORE
#
#     if VECTOR_STORE is not None:
#         print(">>> ✅ Vector Store đã được tải. Bỏ qua khởi tạo.")
#         return
#
#     conn = get_db_connection()
#     if not conn:
#         print("!!! LỖI: Không thể kết nối DB để tải dữ liệu RAG.")
#         return
#
#     try:
#         cursor = conn.cursor(dictionary=True)
#         query = "SELECT ten_benh, dulieubenh FROM benh"
#         cursor.execute(query)
#         data = cursor.fetchall()
#         cursor.close()
#         conn.close()
#
#         texts = []
#         for row in data:
#             # Gộp tên bệnh vào nội dung để làm ngữ cảnh
#             full_text = f"Tên bệnh: {row['ten_benh']}\n\nChi tiết: {row['dulieubenh']}"
#             texts.append(full_text)
#
#         # Chia Chunks (Sử dụng LangChain Text Splitters)
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=80,
#             separators=["\n\n", "\n", ".", "!", "?"]
#         )
#
#         chunks = text_splitter.create_documents(texts)
#
#         # Tạo Embeddings và Vector Store (Sử dụng langchain_chroma)
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         VECTOR_STORE = Chroma.from_documents(chunks, embeddings)
#         print(">>> ✅ Cơ sở dữ liệu Vector RAG (Chroma) đã được khởi tạo thành công.")
#
#     except Exception as e:
#         print(f"!!! LỖI TẠO RAG/VECTOR STORE: {e}")
#         VECTOR_STORE = None
#
#
# # --- CHẠY HÀM KHỞI TẠO RAG KHI ỨNG DỤNG KHỞI ĐỘNG ---
# @app.before_request
# def initialize_rag():
#     """Chạy hàm khởi tạo RAG trước khi xử lý yêu cầu đầu tiên."""
#     if VECTOR_STORE is None:
#         load_and_chunk_data()
#
#
# # =================================================================
# # 2. HÀM XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# # =================================================================
#
# def process_and_predict(base64_img_string):
#     """Xử lý chuỗi base64 thành ảnh, chuẩn hóa và đưa ra dự đoán."""
#     if model is None:
#         return "Lỗi tải mô hình", 0.0
#
#     try:
#         img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
#         img_bytes = base64.b64decode(img_data)
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = x / 255.0  # CHUẨN HÓA
#
#         predictions = model.predict(x)
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
# # --- ROUTE ĐĂNG NHẬP (Login) --- (Giữ nguyên)
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
#             query = "SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s"
#             cursor.execute(query, (taikhoan, mk))
#             user = cursor.fetchone()
#
#             cursor.close()
#             conn.close()
#
#             if user:
#                 session['loggedin'] = True
#                 session['user_id'] = user.get('idTaikhoan')
#                 session['username'] = user.get('taikhoan')
#                 return redirect(url_for('trangchu_page'))
#             else:
#                 error_message = 'Tài khoản hoặc mật khẩu không đúng.'
#         else:
#             error_message = 'Lỗi kết nối cơ sở dữ liệu. Vui lòng kiểm tra lại cấu hình DB.'
#
#     return render_template('login.html', error=error_message)
#
#
# # --- ROUTE TRANG CHỦ & PHÂN LOẠI (Giữ nguyên) ---
# @app.route('/trangchu')
# def trangchu_page():
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('trangchu.html', username=session.get('username'))
#
#
# @app.route('/phan_loai_benh_ga')
# def phan_loai_benh_ga_page():
#     if 'loggedin' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('phan_loai_benh_ga.html', username=session.get('username'))
#
#
# # --- ROUTE XỬ LÝ CHẨN ĐOÁN VÀ KHỞI TẠO CHAT (TÍCH HỢP RAG) ---
# @app.route('/diagnose', methods=['POST'])
# def diagnose_and_start_chat():
#     """Chẩn đoán ảnh, khởi tạo chat session, và trả về phản hồi RAG đầu tiên."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Bạn cần đăng nhập để thực hiện chẩn đoán.'}), 401
#
#     if VECTOR_STORE is None:
#         return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500
#
#     try:
#         # 1. Chẩn đoán ảnh
#         data = request.get_json()
#         img_data_b64 = data.get('image')
#         if not img_data_b64:
#             return jsonify({'error': 'Không tìm thấy dữ liệu ảnh (base64).'}), 400
#
#         predicted_name, confidence = process_and_predict(img_data_b64)
#
#         # Xử lý trường hợp Healthy hoặc Lỗi dự đoán
#         if predicted_name in ["Lỗi xử lý", "Lỗi tải mô hình"]:
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': f"Lỗi hệ thống khi dự đoán."
#             })
#
#         if predicted_name == "Healthy":
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': f"Kết quả chẩn đoán: **{predicted_name}**. Gà của bạn khỏe mạnh, hãy tiếp tục duy trì chế độ chăm sóc tốt!"
#             })
#
#         # 2. Khởi tạo Chatbot và RAG cho bệnh được chẩn đoán
#
#         # a. Truy vấn RAG (Lấy thông tin phác đồ ban đầu)
#         query_rag = f"Phác đồ điều trị ban đầu và triệu chứng chính của bệnh {predicted_name}"
#         rag_docs = VECTOR_STORE.similarity_search(query_rag, k=3)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # b. Thiết lập System Prompt và Message
#         system_prompt = (
#             "Bạn là chuyên gia thú y gia cầm. Hãy cung cấp tư vấn chi tiết DỰA TRÊN NGỮ CẢNH "
#             "CHUYÊN MÔN được cung cấp. Tuyệt đối không tự suy diễn nếu không có dữ liệu. "
#             "Sử dụng Markdown để định dạng câu trả lời dễ đọc."
#         )
#         initial_prompt = (
#             f"Kết quả chẩn đoán hình ảnh là **{predicted_name}** với độ tin cậy {confidence:.2f}%. "
#             f"Dưới đây là các dữ liệu chuyên môn về bệnh này:\n\n"
#             f"--- NGUỒN DỮ LIỆU RAG ---\n{rag_context}\n--- END RAG ---\n\n"
#             "Dựa vào thông tin trên, hãy đưa ra: 1. Tóm tắt nhanh về bệnh. 2. Phác đồ điều trị khẩn cấp ban đầu (thuốc và cách ly)."
#         )
#
#         # c. Khởi tạo Chat Session với System Prompt
#         chat = gemini_client.chats.create(
#             model=LLM_MODEL,
#             config={'system_instruction': system_prompt}
#         )
#
#         # d. Gửi tin nhắn đầu tiên và lưu Session
#         initial_response = chat.send_message(initial_prompt)
#         ACTIVE_CHATS[user_id] = chat  # Lưu lại phiên chat
#
#         # 3. Trả về kết quả JSON và Phản hồi Chatbot
#         return jsonify({
#             'success': True,
#             'prediction': {
#                 'disease': predicted_name,
#                 'confidence': f'{confidence:.2f}%'
#             },
#             'initial_chat_response': initial_response.text
#         })
#
#     except Exception as e:
#         print(f"LỖI KHỞI TẠO CHAT VÀ RAG: {e}")
#         # Xóa chat session nếu lỗi
#         if user_id in ACTIVE_CHATS: del ACTIVE_CHATS[user_id]
#         return jsonify({'error': f'Lỗi hệ thống khi khởi tạo tư vấn: {str(e)}'}), 500
#
#
# # --- ROUTE XỬ LÝ CÂU HỎI TIẾP THEO ---
# @app.route('/chat', methods=['POST'])
# def handle_followup_chat():
#     """API nhận câu hỏi tiếp theo, sử dụng RAG và phiên chat đã lưu."""
#     user_id = session.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Bạn cần đăng nhập.'}), 401
#
#     if user_id not in ACTIVE_CHATS:
#         return jsonify({'error': 'Chưa có phiên chat nào được khởi tạo. Vui lòng chẩn đoán trước.'}), 400
#
#     if VECTOR_STORE is None:
#         return jsonify({'error': 'Hệ thống RAG chưa được khởi tạo thành công.'}), 500
#
#     try:
#         data = request.get_json()
#         user_question = data.get('question')
#
#         if not user_question:
#             return jsonify({'error': 'Không tìm thấy câu hỏi.'}), 400
#
#         # Lấy phiên chat hiện tại
#         current_chat = ACTIVE_CHATS[user_id]
#
#         # 1. Truy vấn RAG (Lấy thông tin liên quan đến câu hỏi mới)
#         rag_docs = VECTOR_STORE.similarity_search(user_question, k=3)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # 2. Gộp RAG vào Prompt
#         augmented_prompt = (
#             f"Dựa trên NGỮ CẢNH BỔ SUNG dưới đây, hãy trả lời câu hỏi: '{user_question}'. "
#             f"Đảm bảo câu trả lời nhất quán với lịch sử trò chuyện (nếu có).\n\n"
#             f"--- NGỮ CẢNH RAG ---\n{rag_context}\n--- END NGỮ CẢNH ---\n"
#         )
#         response = current_chat.send_message(augmented_prompt)
#
#         # 4. Trả về kết quả JSON
#         return jsonify({
#             'success': True,
#             'response': response.text
#         })
#
#     except Exception as e:
#         print(f"LỖI XỬ LÝ CHAT TIẾP THEO: {e}")
#         return jsonify({'error': f'Lỗi server không xác định: {str(e)}'}), 500
#
#     # =================================================================
#     # 4. CHẠY ỨNG DỤNG
#     # =================================================================
#
#     if __name__ == '__main__':
#         app.run(debug=True, host='0.0.0.0', port=5000)
#
#     # 3. Gửi Prompt Tăng cường đến Gemini (giữ lại lịch sử chat)
#
#

#
# import os
# import io
# import re
# import base64
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import mysql.connector
#
# # =================================================================
# # IMPORT LANGCHAIN HIỆN ĐẠI
# # =================================================================
# from google import genai
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
#
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
# except ImportError:
#     print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
#     tf = None
#
# # =================================================================
# # 0. CẤU HÌNH VÀ KHỞI TẠO CHUNG
# # =================================================================
#
# # 🔑 Đảm bảo API Key của bạn còn hoạt động
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCifSb7b1ldIDPiSn7Gz2ZCmTm6HtaLbr0"
# EMBEDDING_MODEL = "text-embedding-004"
# LLM_MODEL = "gemini-2.0-flash"  # Sử dụng model flash mới nhất
#
# try:
#     gemini_client = genai.Client()
# except Exception as e:
#     print(f"!!! LỖI KHỞI TẠO GEMINI CLIENT: {e}")
#
# VECTOR_STORE = None
# ACTIVE_CHATS = {}
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '123456',
#     'database': 'benh_ga'
# }
#
# MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
# CLASS_NAMES = ['Bệnh Cầu Trùng Gà (Coccidiosis)', 'Healthy', 'Bệnh Newcastle (Gà Rù)', 'Salmonella']
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
#
# app = Flask(__name__)
# app.secret_key = 'mot_chuoi_bi_mat_rat_dai_va_kho'
#
# # TẢI MÔ HÌNH
# model = None
# if tf is not None:
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         print(f">>> ✅ Mô hình AI đã tải thành công.")
#     except Exception as e:
#         print(f"!!! LỖI TẢI MÔ HÌNH: {e}")
#
#
# # =================================================================
# # 1. HÀM DATABASE VÀ RAG
# # =================================================================
#
# def get_db_connection():
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         return conn
#     except mysql.connector.Error as err:
#         print(f"Lỗi kết nối database: {err}")
#         return None
#
#
# def load_and_chunk_data():
#     global VECTOR_STORE
#     if VECTOR_STORE is not None: return
#
#     conn = get_db_connection()
#     if not conn: return
#
#     try:
#         cursor = conn.cursor(dictionary=True)
#         query = "SELECT ten_benh, dulieubenh FROM benh"
#         cursor.execute(query)
#         data = cursor.fetchall()
#         cursor.close()
#         conn.close()
#
#         texts = []
#         for row in data:
#             full_text = f"Bệnh: {row['ten_benh']}\nNội dung: {row['dulieubenh']}"
#             texts.append(full_text)
#
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
#         chunks = text_splitter.create_documents(texts)
#
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         VECTOR_STORE = Chroma.from_documents(chunks, embeddings)
#         print(">>> ✅ RAG Vector Store sẵn sàng.")
#     except Exception as e:
#         print(f"!!! LỖI RAG: {e}")
#
#
# @app.before_request
# def initialize_rag():
#     if VECTOR_STORE is None:
#         load_and_chunk_data()
#
#
# # =================================================================
# # 2. XỬ LÝ CHẨN ĐOÁN (ĐÃ SỬA THEO Ý BẠN)
# # =================================================================
#
# def process_and_predict(base64_img_string):
#     if model is None: return "Lỗi tải mô hình", 0.0
#     try:
#         img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
#         img_bytes = base64.b64decode(img_data)
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
#         x = image.img_to_array(img) / 255.0
#         x = np.expand_dims(x, axis=0)
#         predictions = model.predict(x)
#         idx = np.argmax(predictions[0])
#         return CLASS_NAMES[idx], np.max(predictions[0]) * 100
#     except Exception as e:
#         return f"Lỗi: {str(e)}", 0.0
#
#
# @app.route('/diagnose', methods=['POST'])
# def diagnose_and_start_chat():
#     user_id = session.get('user_id')
#     if not user_id: return jsonify({'error': 'Chưa đăng nhập'}), 401
#
#     try:
#         data = request.get_json()
#         predicted_name, confidence = process_and_predict(data.get('image'))
#
#         if predicted_name == "Healthy":
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': 'Khỏe mạnh', 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': "Gà của bạn có vẻ rất khỏe mạnh! Hãy tiếp tục duy trì chế độ dinh dưỡng tốt nhé."
#             })
#
#         # --- PHẦN THAY ĐỔI: HỎI Ý KIẾN TRƯỚC ---
#                 system_prompt = (
#                     "BẠN LÀ CHUYÊN GIA THÚ Y GÀ - TRỢ LÝ ĐẮC LỰC CỦA WEB CHICKEN AI.\n\n"
#
#                     "KỶ LUẬT TRẢ LỜI (NGHIÊM NGẶT):\n"
#                     "1. CHỈ TRẢ LỜI dựa trên thông tin có trong 'DỮ LIỆU THÚ Y' được cung cấp. Tuyệt đối không tự bịa ra kiến thức ngoài.\n"
#                     "2. PHÂN BIỆT BỆNH: Nếu người dùng hỏi về 'Newcastle' hoặc 'Gà rù', CHỈ lấy dữ liệu của Newcastle. Nếu hỏi 'Cầu trùng', CHỈ lấy dữ liệu Cầu trùng. Không được trả lời nhầm nội dung bệnh này cho bệnh kia.\n"
#                     "3. XÁC NHẬN TÊN: Hiểu rằng 'New Castle Disease', 'Newcastle' và 'Gà Rù' là cùng một bệnh.\n"
#                     "4. Nếu thông tin trong Database bị thiếu hoặc là NULL, hãy báo: 'Xin lỗi, hiện tại hệ thống chưa cập nhật chi tiết mục này cho bệnh [Tên bệnh].'\n\n"
#
#                     "QUY ĐỊNH TRÌNH BÀY (ĐỂ KHÔNG BỊ RỐI):\n"
#                     "- TUYỆT ĐỐI KHÔNG sử dụng các ký tự: * (dấu sao), # (dấu thăng), ** (in đậm).\n"
#                     "- SỬ DỤNG CHỮ VIẾT HOA CÓ DẤU cho các tiêu đề mục lớn (Ví dụ: NGUYÊN NHÂN, TRIỆU CHỨNG, ĐIỀU TRỊ).\n"
#                     "- Mỗi ý con bắt buộc phải xuống dòng và bắt đầu bằng dấu gạch ngang (-).\n"
#                     "- GIỮA CÁC MỤC LỚN PHẢI CÁCH NHAU 1 DÒNG TRỐNG (dùng hai dấu xuống dòng \\n\\n).\n"
#                     "- Trình bày theo dạng danh sách, không viết thành một khối văn bản dài dặc.\n\n"
#
#                     "PHONG CÁCH: Chuyên nghiệp, ngắn gọn, đi thẳng vào vấn đề hỗ trợ người chăn nuôi."
#                 )
#
#         # Tạo phiên chat mới
#         chat = gemini_client.chats.create(model=LLM_MODEL, config={'system_instruction': system_prompt})
#         ACTIVE_CHATS[user_id] = chat
#
#         # Câu chào hỏi đầu tiên (Không đưa phác đồ ngay)
#         initial_greeting = (
#             f"Dựa trên hình ảnh, tôi chẩn đoán gà có khả năng cao mắc bệnh **{predicted_name}** ({confidence:.2f}%). "
#             f"\n\nChào bạn, tôi rất tiếc khi biết gà của bạn có dấu hiệu mắc bệnh này. "
#             f"Bạn có muốn tôi giúp bạn tìm hiểu chi tiết về các triệu chứng và cung cấp phác đồ điều trị cho bệnh {predicted_name} ngay không?"
#         )
#
#         return jsonify({
#             'success': True,
#             'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#             'initial_chat_response': initial_greeting
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#
# # =================================================================
# # 3. CHAT TIẾP THEO (KHI NGƯỜI DÙNG ĐỒNG Ý)
# # =================================================================
#
# @app.route('/chat', methods=['POST'])
# def handle_followup_chat():
#     user_id = session.get('user_id')
#     current_chat = ACTIVE_CHATS.get(user_id)
#     if not current_chat: return jsonify({'error': 'Phiên chat hết hạn'}), 400
#
#     try:
#         data = request.get_json()
#         user_question = data.get('question')
#
#         # Truy vấn RAG để lấy kiến thức từ Database
#         rag_docs = VECTOR_STORE.similarity_search(user_question, k=3)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # Gửi kèm ngữ cảnh cho AI
#         full_prompt = (
#             f"Sử dụng thông tin sau để trả lời người dùng: \n{rag_context}\n\n"
#             f"Câu hỏi: {user_question}"
#         )
#         response = current_chat.send_message(full_prompt)
#
#         return jsonify({'success': True, 'response': response.text})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#
# # Các route login/trang chu giữ nguyên của bạn...
# @app.route('/', methods=['GET', 'POST'])
# @app.route('/login', methods=['GET', 'POST'])
# def login_page():
#     if request.method == 'POST':
#         taikhoan, mk = request.form.get('taikhoan'), request.form.get('mk')
#         conn = get_db_connection()
#         if conn:
#             cursor = conn.cursor(dictionary=True)
#             cursor.execute("SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s", (taikhoan, mk))
#             user = cursor.fetchone()
#             conn.close()
#             if user:
#                 session['loggedin'], session['user_id'], session['username'] = True, user['idTaikhoan'], user[
#                     'taikhoan']
#                 return redirect(url_for('trangchu_page'))
#     return render_template('login.html')
#
#
# @app.route('/trangchu')
# def trangchu_page():
#     if 'loggedin' not in session: return redirect(url_for('login_page'))
#     return render_template('trangchu.html', username=session.get('username'))
#
#
# @app.route('/phan_loai_benh_ga')
# def phan_loai_benh_ga_page():
#     if 'loggedin' not in session: return redirect(url_for('login_page'))
#     return render_template('phan_loai_benh_ga.html', username=session.get('username'))
#
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


# import os
# import io
# import re
# import base64
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import mysql.connector
#
# # =================================================================
# # IMPORT AI & RAG TOOLKIT
# # =================================================================
# from google import genai
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
#
# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing import image
# except ImportError:
#     print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
#     tf = None
#
# # =================================================================
# # 0. CẤU HÌNH HỆ THỐNG
# # =================================================================
#
# # 🔑 API Key và Model config
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCifSb7b1ldIDPiSn7Gz2ZCmTm6HtaLbr0"
# EMBEDDING_MODEL = "text-embedding-004"
# LLM_MODEL = "gemini-2.0-flash"
#
# try:
#     gemini_client = genai.Client()
# except Exception as e:
#     print(f"!!! LỖI KHỞI TẠO GEMINI CLIENT: {e}")
#
# VECTOR_STORE = None
# ACTIVE_CHATS = {}
#
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '123456',
#     'database': 'benh_ga'
# }
#
# MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
# # Sửa lại cho khớp với cột ten_benh trong DB
# CLASS_NAMES = ['Bệnh Cầu Trùng Gà (Coccidiosis)', 'Healthy', 'Bệnh Newcastle (Gà Rù)', 'Salmonella']
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
#
# app = Flask(__name__)
# app.secret_key = 'capstone_chicken_ai_key_secret'
#
# # TẢI MÔ HÌNH CHẨN ĐOÁN ẢNH
# model = None
# if tf is not None:
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         print(f">>> ✅ Mô hình AI Chẩn đoán đã sẵn sàng.")
#     except Exception as e:
#         print(f"!!! LỖI TẢI MÔ HÌNH: {e}")
#
# # =================================================================
# # 1. QUẢN LÝ DATABASE & RAG (KỸ THUẬT TỐI ƯU)
# # =================================================================
#
# def get_db_connection():
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)
#         return conn
#     except mysql.connector.Error as err:
#         print(f"Lỗi kết nối database: {err}")
#         return None
#
# def load_and_chunk_data():
#     """Đọc dữ liệu từ 6 cột MySQL và nạp vào Vector Database"""
#     global VECTOR_STORE
#     if VECTOR_STORE is not None: return
#
#     conn = get_db_connection()
#     if not conn: return
#
#     try:
#         cursor = conn.cursor(dictionary=True)
#         # 🟢 TRUY VẤN TẤT CẢ CỘT CHI TIẾT
#         query = "SELECT ten_benh, mo_ta_benh, nguyen_nhan, trieu_chung, phong_benh, dieu_tri_vac_xin FROM benh"
#         cursor.execute(query)
#         data = cursor.fetchall()
#         cursor.close()
#         conn.close()
#
#         texts = []
#         for row in data:
#             # Gắn nhãn rõ ràng để Vector Store tìm kiếm theo ngữ cảnh tốt hơn
#             full_content = (
#                 f"BỆNH: {row['ten_benh']}\n"
#                 f"MÔ TẢ: {row['mo_ta_benh']}\n"
#                 f"NGUYÊN NHÂN: {row['nguyen_nhan']}\n"
#                 f"TRIỆU CHỨNG: {row['trieu_chung']}\n"
#                 f"PHÒNG BỆNH: {row['phong_benh']}\n"
#                 f"ĐIỀU TRỊ & VACCINE: {row['dieu_tri_vac_xin']}"
#             )
#             texts.append(full_content)
#
#         # Chunk size 1200 giúp giữ trọn vẹn thông tin một thể bệnh
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
#         chunks = text_splitter.create_documents(texts)
#
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         VECTOR_STORE = Chroma.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             persist_directory="./chroma_db" # Lưu lại để không phải load nhiều lần
#         )
#         print(">>> ✅ Vector Store đã nạp dữ liệu chi tiết thành công.")
#     except Exception as e:
#         print(f"!!! LỖI RAG: {e}")
#
# @app.before_request
# def initialize_rag():
#     if VECTOR_STORE is None:
#         load_and_chunk_data()
#
# # =================================================================
# # 2. XỬ LÝ CHẨN ĐOÁN VÀ CHAT KHỞI TẠO
# # =================================================================
#
# def process_and_predict(base64_img_string):
#     if model is None: return "Lỗi hệ thống", 0.0
#     try:
#         img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
#         img_bytes = base64.b64decode(img_data)
#         img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
#         x = image.img_to_array(img) / 255.0
#         x = np.expand_dims(x, axis=0)
#         predictions = model.predict(x)
#         idx = np.argmax(predictions[0])
#         return CLASS_NAMES[idx], np.max(predictions[0]) * 100
#     except Exception as e:
#         return f"Lỗi: {str(e)}", 0.0
#
# @app.route('/diagnose', methods=['POST'])
# def diagnose_and_start_chat():
#     user_id = session.get('user_id')
#     if not user_id: return jsonify({'error': 'Vui lòng đăng nhập'}), 401
#
#     try:
#         data = request.get_json()
#         predicted_name, confidence = process_and_predict(data.get('image'))
#
#         if predicted_name == "Healthy":
#             return jsonify({
#                 'success': True,
#                 'prediction': {'disease': 'Khỏe mạnh', 'confidence': f'{confidence:.2f}%'},
#                 'initial_chat_response': "Tuyệt vời! Kết quả cho thấy gà của bạn khỏe mạnh. Hãy duy trì vệ sinh chuồng trại nhé!"
#             })
#
#         # Thiết lập System Instruction cho bác sĩ AI
#         # System prompt mới ép AI không dùng dấu sao
#         system_prompt = (
#             "BẠN LÀ CHUYÊN GIA THÚ Y GÀ - TRỢ LÝ ĐẮC LỰC CỦA WEB CHICKEN AI.\n\n"
#
#             "KỶ LUẬT TRẢ LỜI (NGHIÊM NGẶT):\n"
#             "1. CHỈ TRẢ LỜI dựa trên thông tin có trong 'DỮ LIỆU THÚ Y' được cung cấp. Tuyệt đối không tự bịa ra kiến thức ngoài.\n"
#             "2. PHÂN BIỆT BỆNH: Nếu người dùng hỏi về 'Newcastle' hoặc 'Gà rù', CHỈ lấy dữ liệu của Newcastle. Nếu hỏi 'Cầu trùng', CHỈ lấy dữ liệu Cầu trùng. Không được trả lời nhầm nội dung bệnh này cho bệnh kia.\n"
#             "3. XÁC NHẬN TÊN: Hiểu rằng 'New Castle Disease', 'Newcastle' và 'Gà Rù' là cùng một bệnh.\n"
#             "4. Nếu thông tin trong Database bị thiếu hoặc là NULL, hãy báo: 'Xin lỗi, hiện tại hệ thống chưa cập nhật chi tiết mục này cho bệnh [Tên bệnh].'\n\n"
#
#             "QUY ĐỊNH TRÌNH BÀY (ĐỂ KHÔNG BỊ RỐI):\n"
#             "- TUYỆT ĐỐI KHÔNG sử dụng các ký tự: * (dấu sao), # (dấu thăng), ** (in đậm).\n"
#             "- SỬ DỤNG CHỮ VIẾT HOA CÓ DẤU cho các tiêu đề mục lớn (Ví dụ: NGUYÊN NHÂN, TRIỆU CHỨNG, ĐIỀU TRỊ).\n"
#             "- Mỗi ý con bắt buộc phải xuống dòng và bắt đầu bằng dấu gạch ngang (-).\n"
#             "- GIỮA CÁC MỤC LỚN PHẢI CÁCH NHAU 1 DÒNG TRỐNG (dùng hai dấu xuống dòng \\n\\n).\n"
#             "- Trình bày theo dạng danh sách, không viết thành một khối văn bản dài dặc.\n\n"
#
#             "PHONG CÁCH: Chuyên nghiệp, ngắn gọn, đi thẳng vào vấn đề hỗ trợ người chăn nuôi."
#         )
#
#         chat = gemini_client.chats.create(model=LLM_MODEL, config={'system_instruction': system_prompt})
#         ACTIVE_CHATS[user_id] = chat
#
#         initial_greeting = (
#             f"Tôi chẩn đoán gà có khả năng cao mắc bệnh **{predicted_name}** ({confidence:.2f}%). "
#             f"\n\nChào bạn, đây là một bệnh cần can thiệp sớm. "
#             f"Bạn có muốn tôi liệt kê chi tiết triệu chứng và phác đồ điều trị từ cơ sở dữ liệu cho bạn không?"
#         )
#
#         return jsonify({
#             'success': True,
#             'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
#             'initial_chat_response': initial_greeting
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# # =================================================================
# # 3. HÀM CHAT RAG (TRUY XUẤT THÔNG TIN CHUYÊN SÂU)
# # =================================================================
#
# @app.route('/chat', methods=['POST'])
# def handle_followup_chat():
#     user_id = session.get('user_id')
#     current_chat = ACTIVE_CHATS.get(user_id)
#     if not current_chat: return jsonify({'error': 'Phiên chat đã kết thúc'}), 400
#
#     try:
#         data = request.get_json()
#         user_question = data.get('question')
#
#         # 🔍 TRUY XUẤT RAG (Tìm kiếm thông tin từ 6 cột đã nạp)
#         # Sử dụng similarity_search để bốc ra đúng đoạn mô tả triệu chứng/thuốc
#         rag_docs = VECTOR_STORE.similarity_search(user_question, k=4)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         # Kết hợp câu hỏi người dùng và ngữ cảnh từ DB
#         full_prompt = (
#             f"DỰA VÀO THÔNG TIN CHUYÊN MÔN SAU:\n{rag_context}\n\n"
#             f"CÂU HỎI NGƯỜI DÙNG: {user_question}\n\n"
#             f"HÃY TRẢ LỜI: Dựa hoàn toàn vào dữ liệu trên để tư vấn cho người dùng. "
#             f"Nếu câu hỏi về thuốc hoặc vaccine, hãy liệt kê rõ ràng tên thuốc."
#         )
#         response = current_chat.send_message(full_prompt)
#
#         return jsonify({'success': True, 'response': response.text})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# # =================================================================
# # 4. QUẢN LÝ GIAO DIỆN & LOGIN
# # =================================================================
#
# @app.route('/', methods=['GET', 'POST'])
# @app.route('/login', methods=['GET', 'POST'])
# def login_page():
#     if request.method == 'POST':
#         taikhoan = request.form.get('taikhoan')
#         mk = request.form.get('mk')
#         conn = get_db_connection()
#         if conn:
#             cursor = conn.cursor(dictionary=True)
#             cursor.execute("SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s", (taikhoan, mk))
#             user = cursor.fetchone()
#             conn.close()
#             if user:
#                 session['loggedin'] = True
#                 session['user_id'] = user['idTaikhoan']
#                 session['username'] = user['taikhoan']
#                 return redirect(url_for('trangchu_page'))
#     return render_template('login.html')
#
# @app.route('/trangchu')
# def trangchu_page():
#     if 'loggedin' not in session: return redirect(url_for('login_page'))
#     return render_template('trangchu.html', username=session.get('username'))
#
# @app.route('/phan_loai_benh_ga')
# def phan_loai_benh_ga_page():
#     if 'loggedin' not in session: return redirect(url_for('login_page'))
#     return render_template('phan_loai_benh_ga.html', username=session.get('username'))
#
# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login_page'))
#
# if __name__ == '__main__':
#     # Lưu ý: Chạy host 0.0.0.0 để có thể truy cập từ thiết bị khác trong mạng LAN
#     app.run(debug=True, host='0.0.0.0', port=5000)


import os
import io
import re
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector

# =================================================================
# IMPORT AI & RAG TOOLKIT
# =================================================================
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
except ImportError:
    print("!!! LỖI: Thiếu thư viện AI. Vui lòng cài đặt: pip install tensorflow")
    tf = None

# =================================================================
# 0. CẤU HÌNH HỆ THỐNG
# =================================================================

os.environ["GOOGLE_API_KEY"] = ""
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"

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

MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
# ✅ Tên class khớp 100% với cột ten_benh trong MySQL
# Đảm bảo thứ tự này khớp với thứ tự các Class khi bạn Train Model
# Thứ tự chuẩn để khớp với Label của Model AI
CLASS_NAMES = ['Bệnh Cầu Trùng', 'Gà Khỏe Mạnh', 'Bệnh Gà Rù', 'Bệnh Thương Hàn']
IMG_HEIGHT = 224
IMG_WIDTH = 224

app = Flask(__name__)
app.secret_key = 'capstone_chicken_ai_key_secret'

model = None
if tf is not None:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f">>> ✅ Mô hình AI Chẩn đoán sẵn sàng.")
    except Exception as e:
        print(f"!!! LỖI TẢI MÔ HÌNH: {e}")


# =================================================================
# 1. QUẢN LÝ DATABASE & RAG (CẤU TRÚC 3 CỘT)
# =================================================================

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Lỗi kết nối database: {err}")
        return None


# def load_and_chunk_data():
#     """Đọc dữ liệu từ 3 cột MySQL và nạp vào Vector Database"""
#     global VECTOR_STORE
#     if VECTOR_STORE is not None: return
#
#     conn = get_db_connection()
#     if not conn: return
#
#     try:
#         cursor = conn.cursor(dictionary=True)
#         # 🟢 QUAY LẠI TRUY VẤN 3 CỘT CŨ
#         query = "SELECT ten_benh, dulieubenh FROM benh"
#         cursor.execute(query)
#         data = cursor.fetchall()
#         cursor.close()
#         conn.close()
#
#         texts = []
#         for row in data:
#             full_content = f"BỆNH: {row['ten_benh']}\nNỘI DUNG: {row['dulieubenh']}"
#             texts.append(full_content)
#
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.create_documents(texts)
#
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         VECTOR_STORE = Chroma.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             persist_directory="./chroma_db"
#         )
#         print(">>> ✅ RAG Vector Store (3 cột) đã nạp thành công.")
#     except Exception as e:
#         print(f"!!! LỖI RAG: {e}")
def load_and_chunk_data():
    global VECTOR_STORE
    # Bỏ dòng check None để có thể nạp lại khi cần
    conn = get_db_connection()
    if not conn: return

    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT ten_benh, dulieubenh FROM benh"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        documents = []
        for row in data:
            # ✅ QUAN TRỌNG: Lặp lại tên bệnh ở đầu mỗi đoạn dữ liệu
            # Điều này giúp Vector của "Gà Rù" sẽ khác hẳn Vector của "Cầu Trùng"
            content = f"THÔNG TIN VỀ {row['ten_benh'].upper()}: {row['dulieubenh']}"

            # Chia nhỏ dữ liệu nhưng vẫn giữ ngữ cảnh tên bệnh
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            chunks = text_splitter.split_text(content)

            from langchain_core.documents import Document
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": row['ten_benh']}))

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        VECTOR_STORE = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print(">>> ✅ RAG đã nạp dữ liệu định danh bệnh thành công.")
    except Exception as e:
        print(f"!!! LỖI RAG: {e}")


@app.before_request
def initialize_rag():
    if VECTOR_STORE is None:
        load_and_chunk_data()


# =================================================================
# 2. CHẨN ĐOÁN VÀ CHAT KHỞI TẠO
# =================================================================

def process_and_predict(base64_img_string):
    if model is None: return "Lỗi hệ thống", 0.0
    try:
        img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
        img_bytes = base64.b64decode(img_data)
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)
        idx = np.argmax(predictions[0])
        return CLASS_NAMES[idx], np.max(predictions[0]) * 100
    except Exception as e:
        return f"Lỗi: {str(e)}", 0.0


@app.route('/diagnose', methods=['POST'])
def diagnose_and_start_chat():
    user_id = session.get('user_id')
    if not user_id: return jsonify({'error': 'Vui lòng đăng nhập'}), 401

    try:
        data = request.get_json()
        predicted_name, confidence = process_and_predict(data.get('image'))

        if predicted_name == "Healthy":
            return jsonify({
                'success': True,
                'prediction': {'disease': 'Khỏe mạnh', 'confidence': f'{confidence:.2f}%'},
                'initial_chat_response': "Tuyệt vời! Kết quả cho thấy gà khỏe mạnh. Hãy duy trì vệ sinh chuồng trại nhé!"
            })

        # system_prompt = (
        #     "BẠN LÀ CHUYÊN GIA THÚ Y GÀ - TRỢ LÝ CỦA WEB CHICKEN AI.\n\n"
        #
        #     "KỶ LUẬT TRẢ LỜI:\n"
        #     "1. ƯU TIÊN sử dụng thông tin trong 'DỮ LIỆU THÚ Y'. Nếu dữ liệu bị thiếu một phần, hãy sử dụng kiến thức chuyên môn thú y để bổ sung sao cho chính xác nhất, tuyệt đối không trả lời sai kiến thức y khoa.\n"
        #     "2. NGỮ CẢNH: Hiểu rằng Newcastle và Gà Rù là cùng một bệnh.\n\n"
        #
        #     "QUY ĐỊNH TRÌNH BÀY (GIỮ NGUYÊN Ý BẠN MUỐN):\n"
        #     "- KHÔNG dùng dấu sao (*), dấu thăng (#) hay in đậm (**).\n"
        #     "- VIẾT HOA TOÀN BỘ TIÊU ĐỀ MỤC LỚN (Ví dụ: NGUYÊN NHÂN, TRIỆU CHỨNG).\n"
        #     "- Mỗi ý con bắt đầu bằng dấu gạch ngang (-) và xuống dòng ngay.\n"
        #     "- Khoảng cách: 2 lần xuống dòng giữa các mục lớn."
        # )
        system_prompt = (
            "BẠN LÀ CHUYÊN GIA THÚ Y GÀ - TRỢ LÝ CỦA WEB CHICKEN AI.\n\n"
            "QUY ĐỊNH TRÌNH BÀY (BẮT BUỘC):\n"
            "- Sử dụng dấu chấm tròn (•) hoặc dấu gạch ngang (-) cho danh sách.\n"
            "- Sau mỗi dấu (•) hoặc (-), phải có một dấu cách và BẮT BUỘC xuống dòng ngay lập tức.\n"
            "- Các mục tiêu đề lớn phải VIẾT HOA và cách đoạn bên dưới 1 dòng trống.\n"
            "- Tuyệt đối không viết hoa toàn bộ văn bản nội dung.\n"
            "- Không sử dụng ký tự đặc biệt như * hoặc #."
            "KỶ LUẬT TRẢ LỜI:\n"
            "1. TRUY XUẤT DỮ LIỆU: Bạn phải ưu tiên tuyệt đối thông tin được cung cấp từ Database (RAG). Đây là nguồn kiến thức chuẩn cho hệ thống này.\n"
            "2. KHÔNG TỪ CHỐI: Tuyệt đối không trả lời 'không có dữ liệu' hoặc 'tôi không biết'. Nếu Database thiếu một vài chi tiết nhỏ, hãy sử dụng kiến thức thú y chuyên môn để bổ sung và hướng dẫn bà con đầy đủ, tận tâm.\n"
            # Thêm dòng này vào cuối system_prompt của bạn
            "TUYỆT ĐỐI KHÔNG lấy thông tin điều trị của bệnh Cầu Trùng để trả lời cho bệnh Gà Rù và ngược lại. "
            "Mỗi bệnh có phác đồ khác nhau hoàn toàn: Gà Rù dùng vaccine/kháng thể, Cầu Trùng dùng thuốc trị ký sinh trùng."

            "QUY ĐỊNH TRÌNH BÀY:\n"
            "- KHÔNG VIẾT HOA TOÀN BỘ VĂN BẢN (Để người dân dễ đọc, tránh cảm giác cục súc).\n"
            "- TIÊU ĐỀ MỤC: Viết hoa có dấu và nằm riêng một dòng (Ví dụ: PHÁC ĐỒ ĐIỀU TRỊ, TRIỆU CHỨNG LÂM SÀNG).\n"
            "- HÌNH THỨC: Sử dụng dấu gạch ngang (-) cho các ý con, tuyệt đối không dùng *, #, **.\n"
            "- KHOẢNG CÁCH: Luôn xuống dòng 2 lần giữa các mục lớn để giao diện chat thoáng đãng.\n"
            "- PHONG CÁCH: Chuyên nghiệp, ngắn gọn nhưng phải đầy đủ các bước xử lý chuồng trại và thuốc men.\n"
        )

        chat = gemini_client.chats.create(model=LLM_MODEL, config={'system_instruction': system_prompt})
        ACTIVE_CHATS[user_id] = chat

        initial_greeting = (
            f"Kết quả: Gà có khả năng mắc {predicted_name} ({confidence:.2f}%). "
            f"\n\nChào bạn, tôi là bác sĩ AI. Bạn có muốn tìm hiểu chi tiết triệu chứng và cách điều trị bệnh {predicted_name} ngay không?"
        )

        return jsonify({
            'success': True,
            'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
            'initial_chat_response': initial_greeting
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =================================================================
# 3. CHAT TIẾP THEO (RAG)
# =================================================================

# @app.route('/chat', methods=['POST'])
# def handle_followup_chat():
#     user_id = session.get('user_id')
#     current_chat = ACTIVE_CHATS.get(user_id)
#     if not current_chat: return jsonify({'error': 'Phiên chat hết hạn'}), 400
#
#     try:
#         data = request.get_json()
#         user_question = data.get('question')
#
#         # Truy vấn RAG từ cột dulieubenh
#         rag_docs = VECTOR_STORE.similarity_search(user_question, k=10)
#         rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
#
#         full_prompt = (
#             f"DỮ LIỆU THÚ Y:\n{rag_context}\n\n"
#             f"CÂU HỎI: {user_question}\n\n"
#             "YÊU CẦU: Dựa vào dữ liệu trên để trả lời. Trình bày rõ ràng, không dùng dấu sao, xuống dòng sau mỗi ý."
#         )
#         response = current_chat.send_message(full_prompt)
#
#         return jsonify({'success': True, 'response': response.text})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def handle_followup_chat():
    user_id = session.get('user_id')
    current_chat = ACTIVE_CHATS.get(user_id)
    if not current_chat: return jsonify({'error': 'Phiên chat hết hạn'}), 400

    try:
        data = request.get_json()
        user_question = data.get('question')

        rag_docs = VECTOR_STORE.similarity_search(user_question, k=5)
        rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])

        full_prompt = (
            f"Bối cảnh dữ liệu từ Database:\n{rag_context}\n\n"
            f"Câu hỏi của người dân: {user_question}\n\n"
            "YÊU CẦU: Trình bày câu trả lời rõ ràng. "
            "Sau mỗi dấu gạch ngang (-) bắt đầu ý mới, BẮT BUỘC phải xuống dòng. "
            "Không dùng dấu sao (*)."
        )

        response = current_chat.send_message(full_prompt)

        # Tìm bất kỳ dấu gạch ngang nào đứng sau một ký tự (không phải đầu dòng) và thêm xuống dòng
        clean_response = re.sub(r'([^\n])\s*-\s+', r'\1\n- ', response.text)

        # Xử lý thêm các dấu chấm dính liền với dấu gạch ngang
        clean_response = clean_response.replace(". -", ".\n- ").replace("; -", ";\n- ")

        return jsonify({'success': True, 'response': clean_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Các route giao diện giữ nguyên...
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        taikhoan, mk = request.form.get('taikhoan'), request.form.get('mk')
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s", (taikhoan, mk))
            user = cursor.fetchone()
            conn.close()
            if user:
                session['loggedin'], session['user_id'], session['username'] = True, user['idTaikhoan'], user[
                    'taikhoan']
                return redirect(url_for('trangchu_page'))
    return render_template('login.html')


@app.route('/logout')
def logout_page():
    session.clear()  # Xóa hết dữ liệu phiên đăng nhập
    return redirect(url_for('trangchu_page'))


@app.route('/trangchu')
def trangchu_page():
    if 'loggedin' not in session: return redirect(url_for('login_page'))
    return render_template('trangchu.html', username=session.get('username'))


@app.route('/phan_loai_benh_ga')
def phan_loai_benh_ga_page():
    if 'loggedin' not in session: return redirect(url_for('login_page'))
    return render_template('phan_loai_benh_ga.html', username=session.get('username'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
