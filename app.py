"""
Ứng dụng Backend Flask - Hệ thống hỗ trợ dự đoán nguy cơ tiểu đường
Chức năng:
  - Cung cấp giao diện Web (Trang chủ, Dự đoán, Kết quả, Giới thiệu mô hình).
  - Cung cấp RESTful API (/api/predict, /api/metrics).
  - Nạp mô hình Machine Learning đóng gói (Pipeline) từ models/diabetes_model.pkl một lần duy nhất.
"""

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

# 1. NẠP MÔ HÌNH VÀ THÔNG SỐ ĐÁNH GIÁ MỘT LẦN DUY NHẤT LÚC KHỞI ĐỘNG
print("[SERVER] Đang nạp mô hình Machine Learning...")
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"[SERVER] Đã nạp thành công mô hình từ: {MODEL_PATH}")
except Exception as e:
    model_pipeline = None
    print(f"[SERVER LỖI] Không thể nạp mô hình: {e}")

try:
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
except Exception as e:
    metrics_data = {}
    print(f"[SERVER CẢNH BÁO] Không thể đọc metrics.json: {e}")

# Danh sách 3 đặc trưng đầu vào
FEATURE_NAMES = ['Glucose', 'BMI', 'Age']


def validate_and_parse_input(glucose_val, bmi_val, age_val):
    """
    Hàm kiểm tra và chuẩn hóa dữ liệu nhập vào từ người dùng.
    Trả về: (dict_values, error_message)
    """
    try:
        glucose = float(glucose_val)
        bmi = float(bmi_val)
        age = float(age_val)
    except (ValueError, TypeError):
        return None, "Vui lòng nhập đầy đủ các chỉ số dưới dạng số hợp lệ."

    # Kiểm tra khoảng giá trị sinh lý hợp lý
    if glucose < 0 or glucose > 500:
        return None, "Chỉ số Glucose phải nằm trong khoảng từ 0 đến 500 mg/dL."
    if bmi < 0 or bmi > 100:
        return None, "Chỉ số BMI phải nằm trong khoảng từ 0 đến 100 kg/m²."
    if age < 1 or age > 125:
        return None, "Độ tuổi phải nằm trong khoảng từ 1 đến 125 tuổi."

    return {
        'Glucose': glucose,
        'BMI': bmi,
        'Age': age
    }, None


def generate_medical_interpretation(glucose, bmi, age, prediction, prob_positive):
    """
    Hàm sinh phân tích ý nghĩa lâm sàng và khuyến nghị y tế dựa trên các chỉ số.
    """
    interpretations = []

    # 1. Đánh giá Glucose (Đường huyết lúc đói tham khảo)
    if glucose == 0:
        interpretations.append({
            'feature': 'Glucose (Đường huyết)',
            'value': 'Chưa đo (0 mg/dL)',
            'status': 'warning',
            'note': 'Hệ thống đã tự động ước lượng theo giá trị trung vị của tập dữ liệu y khoa.'
        })
    elif glucose < 100:
        interpretations.append({
            'feature': 'Glucose (Đường huyết)',
            'value': f'{glucose:.1f} mg/dL',
            'status': 'normal',
            'note': 'Trong ngưỡng an toàn bình thường (dưới 100 mg/dL).'
        })
    elif glucose <= 125:
        interpretations.append({
            'feature': 'Glucose (Đường huyết)',
            'value': f'{glucose:.1f} mg/dL',
            'status': 'warning',
            'note': 'Ở mức Tiền đái tháo đường (100 - 125 mg/dL). Cần chú ý điều chỉnh chế độ ăn uống.'
        })
    else:
        interpretations.append({
            'feature': 'Glucose (Đường huyết)',
            'value': f'{glucose:.1f} mg/dL',
            'status': 'danger',
            'note': 'Đường huyết ở mức cao (≥ 126 mg/dL). Đây là dấu hiệu cảnh báo nguy cơ tiểu đường rõ rệt.'
        })

    # 2. Đánh giá BMI (Chỉ số khối cơ thể theo chuẩn WHO châu Á)
    if bmi == 0:
        interpretations.append({
            'feature': 'BMI (Thể trạng)',
            'value': 'Chưa đo (0 kg/m²)',
            'status': 'warning',
            'note': 'Hệ thống đã tự động điền khuyết theo trung vị thể trạng dân số mẫu.'
        })
    elif bmi < 18.5:
        interpretations.append({
            'feature': 'BMI (Thể trạng)',
            'value': f'{bmi:.1f} kg/m²',
            'status': 'normal',
            'note': 'Thể trạng nhẹ cân/gầy so với chiều cao.'
        })
    elif bmi < 23.0:
        interpretations.append({
            'feature': 'BMI (Thể trạng)',
            'value': f'{bmi:.1f} kg/m²',
            'status': 'normal',
            'note': 'Cân nặng chuẩn lý tưởng (18.5 - 22.9 kg/m² theo chuẩn WHO Châu Á).'
        })
    elif bmi < 25.0:
        interpretations.append({
            'feature': 'BMI (Thể trạng)',
            'value': f'{bmi:.1f} kg/m²',
            'status': 'warning',
            'note': 'Tiền béo phì / Thừa cân (23.0 - 24.9 kg/m²). Tăng nguy cơ kháng insulin.'
        })
    else:
        interpretations.append({
            'feature': 'BMI (Thể trạng)',
            'value': f'{bmi:.1f} kg/m²',
            'status': 'danger',
            'note': f'Béo phì ({bmi:.1f} kg/m²). Lượng mỡ thừa là yếu tố thúc đẩy đái tháo đường typ 2.'
        })

    # 3. Đánh giá Độ tuổi
    if age < 35:
        interpretations.append({
            'feature': 'Độ tuổi',
            'value': f'{int(age)} tuổi',
            'status': 'normal',
            'note': 'Nhóm tuổi trẻ, chuyển hóa cơ bản thường hoạt động tốt hơn.'
        })
    elif age < 50:
        interpretations.append({
            'feature': 'Độ tuổi',
            'value': f'{int(age)} tuổi',
            'status': 'warning',
            'note': 'Lứa tuổi trung niên (35 - 49), bắt đầu tích lũy nguy cơ rối loạn chuyển hóa.'
        })
    else:
        interpretations.append({
            'feature': 'Độ tuổi',
            'value': f'{int(age)} tuổi',
            'status': 'warning',
            'note': 'Từ 50 tuổi trở lên là nhóm có nguy cơ tự nhiên cao hơn, cần sàng lọc đường huyết định kỳ.'
        })

    # Khuyến nghị y tế tham khảo
    recommendations = []
    if prediction == 1:
        recommendations.append("Nên sớm đến các cơ sở y tế hoặc bệnh viện để thực hiện xét nghiệm HbA1c và kiểm tra đường huyết tĩnh mạch chính xác.")
        recommendations.append("Hạn chế tối đa nước ngọt có gas, bánh kẹo ngọt và các loại tinh bột tinh chế (cơm trắng, bánh mì trắng).")
        recommendations.append("Tăng cường chất xơ từ rau xanh, các loại hạt và duy trì hoạt động thể chất tối thiểu 30 phút mỗi ngày.")
        recommendations.append("Nếu chỉ số BMI cao, việc giảm 5-10% trọng lượng cơ thể sẽ cải thiện rõ rệt độ nhạy của insulin.")
    else:
        recommendations.append("Duy trì lối sống lành mạnh, cân bằng dinh dưỡng và thói quen vận động đều đặn.")
        recommendations.append("Khám sức khỏe tổng quát định kỳ mỗi 6 - 12 tháng để theo dõi các chỉ số sinh hóa máu.")
        recommendations.append("Uống đủ nước, ngủ đủ giấc và kiểm soát căng thẳng (stress) để duy trì sự ổn định của hệ nội tiết.")

    return interpretations, recommendations


# ==========================================
# CÁC ROUTE PHỤC VỤ GIAO DIỆN WEB
# ==========================================

@app.route('/')
def home():
    """Trang chủ: Giới thiệu hệ thống, các chỉ số và nút bắt đầu"""
    best_acc = 79.87
    if metrics_data and 'models_comparison' in metrics_data:
        best_name = metrics_data.get('best_model', 'Decision Tree')
        best_acc = metrics_data['models_comparison'].get(best_name, {}).get('accuracy', 0.7987) * 100
    return render_template('index.html', best_accuracy=f"{best_acc:.1f}")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Trang nhập thông tin và xử lý dự đoán"""
    if request.method == 'GET':
        return render_template('predict.html', error=None, form_data={})

    # Xử lý form POST từ web
    glucose_raw = request.form.get('glucose', '').strip()
    bmi_raw = request.form.get('bmi', '').strip()
    age_raw = request.form.get('age', '').strip()

    form_data = {'glucose': glucose_raw, 'bmi': bmi_raw, 'age': age_raw}
    clean_vals, err = validate_and_parse_input(glucose_raw, bmi_raw, age_raw)
    if err:
        return render_template('predict.html', error=err, form_data=form_data)

    if model_pipeline is None:
        return render_template('predict.html', error="Hệ thống chưa tải được mô hình. Vui lòng thử lại sau.", form_data=form_data)

    # Chuyển đổi 0 thành NaN để SimpleImputer tự động xử lý
    glucose_in = np.nan if clean_vals['Glucose'] == 0 else clean_vals['Glucose']
    bmi_in = np.nan if clean_vals['BMI'] == 0 else clean_vals['BMI']
    age_in = clean_vals['Age']

    input_df = pd.DataFrame([[glucose_in, bmi_in, age_in]], columns=FEATURE_NAMES)

    # Dự đoán bằng Pipeline đã lưu (tự động Impute -> Scale -> Predict)
    pred_label = int(model_pipeline.predict(input_df)[0])
    pred_probas = model_pipeline.predict_proba(input_df)[0]
    prob_neg = float(pred_probas[0] * 100)
    prob_pos = float(pred_probas[1] * 100)

    interpretations, recommendations = generate_medical_interpretation(
        clean_vals['Glucose'], clean_vals['BMI'], clean_vals['Age'], pred_label, prob_pos
    )

    result_data = {
        'prediction': pred_label,
        'label_text': 'Cảnh báo: Có nguy cơ mắc bệnh tiểu đường' if pred_label == 1 else 'Nguy cơ thấp: Không có dấu hiệu tiểu đường',
        'prob_negative': f"{prob_neg:.1f}",
        'prob_positive': f"{prob_pos:.1f}",
        'confidence': f"{prob_pos if pred_label == 1 else prob_neg:.1f}",
        'glucose': clean_vals['Glucose'],
        'bmi': clean_vals['BMI'],
        'age': int(clean_vals['Age']),
        'interpretations': interpretations,
        'recommendations': recommendations
    }

    return render_template('result.html', result=result_data)


@app.route('/about')
def about():
    """Trang giới thiệu chi tiết mô hình, bảng so sánh và các biểu đồ"""
    return render_template('about.html', metrics=metrics_data)


# ==========================================
# CÁC ROUTE PHỤC VỤ RESTful API (JSON)
# ==========================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API tiếp nhận JSON:
      Request: { "glucose": 130.0, "bmi": 28.5, "age": 45 }
      Response: JSON chứa dự đoán, xác suất, phân tích và khuyến nghị
    """
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Yêu cầu định dạng Content-Type: application/json'}), 400

    data = request.get_json()
    glucose_val = data.get('glucose')
    bmi_val = data.get('bmi')
    age_val = data.get('age')

    clean_vals, err = validate_and_parse_input(glucose_val, bmi_val, age_val)
    if err:
        return jsonify({'status': 'error', 'message': err}), 400

    if model_pipeline is None:
        return jsonify({'status': 'error', 'message': 'Mô hình chưa được nạp trên máy chủ.'}), 500

    glucose_in = np.nan if clean_vals['Glucose'] == 0 else clean_vals['Glucose']
    bmi_in = np.nan if clean_vals['BMI'] == 0 else clean_vals['BMI']
    age_in = clean_vals['Age']

    input_df = pd.DataFrame([[glucose_in, bmi_in, age_in]], columns=FEATURE_NAMES)
    pred_label = int(model_pipeline.predict(input_df)[0])
    pred_probas = model_pipeline.predict_proba(input_df)[0]
    prob_neg = float(pred_probas[0] * 100)
    prob_pos = float(pred_probas[1] * 100)

    interpretations, recommendations = generate_medical_interpretation(
        clean_vals['Glucose'], clean_vals['BMI'], clean_vals['Age'], pred_label, prob_pos
    )

    return jsonify({
        'status': 'success',
        'input': {
            'glucose': clean_vals['Glucose'],
            'bmi': clean_vals['BMI'],
            'age': clean_vals['Age']
        },
        'prediction': pred_label,
        'label': 'Nguy cơ tiểu đường (Dương tính)' if pred_label == 1 else 'Không tiểu đường (Âm tính)',
        'probabilities': {
            'negative_class_0_pct': round(prob_neg, 2),
            'positive_class_1_pct': round(prob_pos, 2)
        },
        'confidence_score_pct': round(prob_pos if pred_label == 1 else prob_neg, 2),
        'interpretations': interpretations,
        'recommendations': recommendations,
        'disclaimer': 'Kết quả từ hệ thống chỉ mang tính chất tham khảo học tập, không thay thế chẩn đoán y khoa từ bác sĩ.'
    }), 200


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API trả về các chỉ số đánh giá thực tế của 3 mô hình từ metrics.json"""
    return jsonify({
        'status': 'success',
        'data': metrics_data
    }), 200


if __name__ == '__main__':
    # Chạy server ở chế độ debug port 5000
    print("[SERVER] Ứng dụng Flask đang khởi chạy tại http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)

