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

def load_and_chunk_data():
    """Đọc dữ liệu từ 3 cột MySQL và nạp vào Vector Database với định danh bệnh"""
    global VECTOR_STORE
    conn = get_db_connection()
    if not conn: return

    try:
        cursor = conn.cursor(dictionary=True)
        # Truy vấn lấy cả tên bệnh và nội dung
        query = "SELECT ten_benh, dulieubenh FROM benh"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        documents = []
        for row in data:
            # ✅ FIX 1: Dán tên bệnh vào nội dung để Vector hóa chính xác
            content = f"THÔNG TIN VỀ {row['ten_benh'].upper()}: {row['dulieubenh']}"

            # ✅ FIX 2: Chia nhỏ văn bản (Chunking) theo Token/Ký tự
            # chunk_size=600 là vừa đủ một ý chính, overlap=50 giúp không mất ngữ cảnh
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            chunks = text_splitter.split_text(content)

            from langchain_core.documents import Document
            for chunk in chunks:
                # Lưu kèm metadata để sau này có thể lọc nếu cần
                documents.append(Document(page_content=chunk, metadata={"source": row['ten_benh']}))

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        # ✅ FIX 3: Lưu vào ChromaDB (Vector Database)
        VECTOR_STORE = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print(">>> ✅ Hệ thống RAG đã được cập nhật dữ liệu định danh bệnh.")
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


        system_prompt = (
            "BẠN LÀ CHUYÊN GIA THÚ Y GÀ - TRỢ LÝ ĐẮC LỰC CỦA WEB CHICKEN AI.\n\n"

            "KỶ LUẬT TRẢ LỜI (TRỊNH TRỌNG):\n"
            "1. TUYỆT ĐỐI không lấy râu ông nọ chắp cằm bà kia. Bệnh Gà Rù (Newcastle) và Cầu Trùng có phác đồ khác biệt hoàn toàn, không được nhầm lẫn.\n"
            "2. ƯU TIÊN thông tin từ Database (RAG). Nếu thiếu, hãy dùng kiến thức chuyên môn để bổ sung đầy đủ, không được từ chối trả lời bà con.\n\n"

            "QUY ĐỊNH TRÌNH BÀY (ĐỂ GIAO DIỆN SẠCH GỌN):\n"
            "- TIÊU ĐỀ MỤC: Viết hoa có dấu, nằm riêng 1 dòng. Cách đoạn dưới 1 dòng trống.\n"
            "- DANH SÁCH: Mỗi ý con bắt đầu bằng dấu gạch ngang (-). \n"
            "- XUỐNG DÒNG: Sau mỗi dấu (-) BẮT BUỘC phải xuống dòng ngay lập tức. Mỗi dòng chỉ chứa 1 thông tin.\n"
            "- KHOẢNG CÁCH: Xuống dòng 2 lần giữa các mục lớn (Ví dụ: TRIỆU CHỨNG và PHÁC ĐỒ).\n"
            "- CẤM: Không dùng các ký tự *, #, ** hoặc dấu chấm tròn (•) để tránh lỗi hiển thị HTML.\n"
            "- PHONG CÁCH: Thân thiện với nông dân, chuyên nghiệp, ngắn gọn, rõ ràng."
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
