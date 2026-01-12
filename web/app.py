import os
import io
import re
import base64
import numpy as np
import mysql.connector
import time
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

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

FILTER_MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\Classify_model.keras'
MODEL_PATH = r'D:\Hoc Ki Cuoi\Web_Chicken\web\model\best_model.keras'
CLASS_NAMES = ['Bệnh Cầu Trùng', 'Gà Khỏe Mạnh', 'Bệnh Gà Rù', 'Bệnh Thương Hàn']
FILTER_CLASSES = ['No Poop', 'Poop']
IMG_HEIGHT, IMG_WIDTH = 224, 224

app = Flask(__name__)
app.secret_key = 'capstone_chicken_ai_key_secret'

filter_model = model = None
if tf is not None:
    try:
        filter_model = tf.keras.models.load_model(FILTER_MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        print(">>> ✅ Hệ thống AI (Lọc & Chẩn đoán) sẵn sàng.")
    except Exception as e:
        print(f"!!! LỖI TẢI MÔ HÌNH: {e}")


# =================================================================
# 1. HÀM HỖ TRỢ XỬ LÝ VĂN BẢN & AI
# =================================================================

def format_clean_text(text):
    """Xử lý triệt để dấu sao và lỗi giãn dòng quá mức."""
    # Xóa sạch các dấu sao và thăng định dạng Markdown
    text = text.replace("**", "").replace("*", "").replace("#", "")

    # Chuẩn hóa khoảng trắng đầu/cuối dòng
    lines = [line.strip() for line in text.split('\n')]

    # Loại bỏ các dòng trống dư thừa để nén văn bản gọn gàng
    compact_lines = []
    for line in lines:
        if line or (compact_lines and compact_lines[-1]):
            compact_lines.append(line)

    return '\n'.join(compact_lines)


def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error:
        return None


def is_it_poop(base64_img_string):
    if filter_model is None: return True
    try:
        img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
        img_bytes = base64.b64decode(img_data)
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        return FILTER_CLASSES[np.argmax(filter_model.predict(x)[0])] == 'Poop'
    except:
        return False


def process_and_predict(base64_img_string):
    if model is None: return "Lỗi hệ thống", 0.0
    try:
        img_data = re.sub('^data:image/.+;base64,', '', base64_img_string)
        img_bytes = base64.b64decode(img_data)
        img = image.load_img(io.BytesIO(img_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        idx = np.argmax(preds[0])
        return CLASS_NAMES[idx], np.max(preds[0]) * 100
    except:
        return "Lỗi", 0.0


# =================================================================
# 2. CHẨN ĐOÁN & LIVE-WEB RAG
# =================================================================

@app.route('/diagnose', methods=['POST'])
def diagnose_and_start_chat():
    user_id = session.get('user_id')
    if not user_id: return jsonify({'error': 'Vui lòng đăng nhập'}), 401

    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64: return jsonify({'error': 'Thiếu ảnh'}), 400

        if not is_it_poop(image_base64):
            return jsonify({'success': False, 'error': 'Đây không phải ảnh phân gà. Hãy chụp lại!'})

        predicted_name, confidence = process_and_predict(image_base64)

        # Lưu lịch sử database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                img_data = re.sub('^data:image/.+;base64,', '', image_base64)
                filename = f"chandoan_{user_id}_{int(time.time())}.jpg"
                if not os.path.exists('static/uploads'): os.makedirs('static/uploads')
                with open(os.path.join('static/uploads', filename), 'wb') as f:
                    f.write(base64.b64decode(img_data))
                sql = "INSERT INTO lich_su_chan_doan (idTaikhoan, ten_benh, do_tin_cay, duong_dan_anh, ngay_tao) VALUES (%s, %s, %s, %s, NOW())"
                cursor.execute(sql, (user_id, predicted_name, float(confidence / 100), filename))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as db_e:
                print(f"Lỗi DB: {db_e}")

        # Định nghĩa URL nguồn dữ liệu (Live-Web RAG)
        reference_urls = []
        if predicted_name == 'Bệnh Cầu Trùng':
            reference_urls = [
                "https://goovetvn.com/tin/benh-cau-trung-o-ga-va-phac-do-dieu-tri-hieu-qua-nhat.html",
                "https://chauthanhjsc.com.vn/phac-do-dieu-tri-cau-trung-o-ga",
                "https://goovetvn.com/benh/benh-cau-trung-o-ga-va-cach-dieu-tri.html"
            ]
        elif predicted_name == 'Bệnh Gà Rù':
            reference_urls = [
                "https://goovetvn.com/tin/cach-tri-benh-ga-ru-bang-toi.html",
                "https://khoathuy.vnua.edu.vn/4036.html",
                "https://goovetvn.com/benh/benh-newcastle-tren-ga.html"
            ]
        elif predicted_name == 'Bệnh Thương Hàn':
            reference_urls = [
                "https://khoathuy.vnua.edu.vn/4036.html",
                "https://goovetvn.com/benh/benh-thuong-han-o-ga-va-phac-do-dieu-tri.html",
                "https://tantienvet.com/benh-thuong-han-ga-va-cach-phong-tri.html"
            ]

        if predicted_name == "Gà Khỏe Mạnh":
            return jsonify({
                'success': True,
                'prediction': {'disease': 'Khỏe mạnh', 'confidence': f'{confidence:.2f}%'},
                'initial_chat_response': "Tuyệt vời! Kết quả cho thấy gà khỏe mạnh. Hãy duy trì vệ sinh tốt nhé!"
            })

        # Cấu hình AI chuyên gia
        system_prompt = (
            f"BẠN LÀ CHUYÊN GIA THÚ Y GÀ CỦA WEB CHICKEN AI.\n"
            f"ĐỐI TƯỢNG: Người nuôi gà (xưng hô là 'bạn'). KHÔNG dùng 'bà con'.\n"
            f"NHIỆM VỤ: Tổng hợp thông tin từ các URL để tư vấn về {predicted_name}.\n"
            "YÊU CẦU ĐỊNH DẠNG:\n"
            "1. TUYỆT ĐỐI KHÔNG dùng dấu sao (*), dấu thăng (#) hoặc Markdown.\n"
            "2. TIÊU ĐỀ MỤC: Viết hoa có dấu, đứng riêng một dòng.\n"
            "3. TRÌNH BÀY GỌN: Không xuống dòng dư thừa. Xuống dòng sau gạch ngang (-)."
        )

        chat = gemini_client.chats.create(model=LLM_MODEL, config={'system_instruction': system_prompt})
        ACTIVE_CHATS[user_id] = chat

        initial_query = [
            f"Chẩn đoán: {predicted_name} ({confidence:.2f}%). Hãy chào bạn ấy và tổng hợp phác đồ điều trị chi tiết từ TẤT CẢ nguồn này:",
            *reference_urls
        ]

        response = chat.send_message(initial_query)

        # Làm sạch và nén văn bản trước khi gửi
        clean_text = format_clean_text(response.text)

        return jsonify({
            'success': True,
            'prediction': {'disease': predicted_name, 'confidence': f'{confidence:.2f}%'},
            'initial_chat_response': clean_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def handle_followup_chat():
    user_id = session.get('user_id')
    current_chat = ACTIVE_CHATS.get(user_id)

    if not current_chat:
        system_prompt = "BẠN LÀ CHUYÊN GIA THÚ Y GÀ. Xưng hô là 'bạn'. Không dùng dấu sao."
        current_chat = gemini_client.chats.create(model=LLM_MODEL, config={'system_instruction': system_prompt})
        ACTIVE_CHATS[user_id] = current_chat

    try:
        data = request.get_json()
        user_question = data.get('question')
        response = current_chat.send_message(user_question)

        # Áp dụng bộ lọc nén dòng và xóa dấu sao
        clean_response = format_clean_text(response.text)

        return jsonify({'success': True, 'response': clean_response})
    except Exception as e:
        return jsonify({'error': "Hệ thống bận, hãy thử lại!"}), 500


# =================================================================
# 3. QUẢN LÝ GIAO DIỆN & TÀI KHOẢN
# =================================================================

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        tk, mk = request.form.get('taikhoan'), request.form.get('mk')
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT idTaikhoan, taikhoan FROM user WHERE taikhoan = %s AND matkhau = %s", (tk, mk))
            user = cursor.fetchone()
            conn.close()
            if user:
                session.update({'loggedin': True, 'user_id': user['idTaikhoan'], 'username': user['taikhoan']})
                return redirect(url_for('trangchu_page'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        tk, mk = request.form.get('taikhoan'), request.form.get('mk')
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO user (taikhoan, matkhau) VALUES (%s, %s)", (tk, mk))
            conn.commit()
            conn.close()
            return render_template('register.html', success="Đăng ký thành công!")
    return render_template('register.html')


@app.route('/logout')
def logout_page():
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/trangchu')
def trangchu_page():
    if 'loggedin' not in session: return redirect(url_for('login_page'))
    return render_template('trangchu.html', username=session.get('username'))


@app.route('/phan_loai_benh_ga')
def phan_loai_benh_ga_page():
    if 'loggedin' not in session: return redirect(url_for('login_page'))
    return render_template('phan_loai_benh_ga.html', username=session.get('username'))


@app.route('/lich-su')
def lich_su_page():
    user_id = session.get('user_id')
    if not user_id: return redirect(url_for('login_page'))
    conn, ds = get_db_connection(), []
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM lich_su_chan_doan WHERE idTaikhoan = %s ORDER BY ngay_tao DESC", (user_id,))
        ds = cursor.fetchall()
        conn.close()
    return render_template('lich_su.html', lich_su=ds, username=session.get('username'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)