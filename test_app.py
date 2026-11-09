"""
Kiểm thử toàn diện ứng dụng Flask và các Route API
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
from app import app

client = app.test_client()

print("=" * 60)
print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN CÁC ROUTE FLASK")
print("=" * 60)

# 1. Kiểm thử Trang chủ GET /
res = client.get('/')
print(f"1. GET /: Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
assert "DiaPredict" in res.get_data(as_text=True)

# 2. Kiểm thử Trang Dự đoán GET /predict
res = client.get('/predict')
print(f"2. GET /predict: Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
assert "Phiếu Thông Tin Chỉ Số" in res.get_data(as_text=True)

# 3. Kiểm thử POST /predict với dữ liệu hợp lệ (Người có nguy cơ)
res = client.post('/predict', data={'glucose': '165', 'bmi': '35.5', 'age': '50'})
print(f"3. POST /predict (Nguy cơ cao): Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
html_text = res.get_data(as_text=True)
assert "Báo Cáo Sàng Lọc Nguy Cơ Sức Khỏe" in html_text
assert "Độ tin cậy của mô hình" in html_text

# 4. Kiểm thử POST /predict với dữ liệu hợp lệ (Người khỏe mạnh)
res = client.post('/predict', data={'glucose': '85', 'bmi': '21.0', 'age': '24'})
print(f"4. POST /predict (Nguy cơ thấp): Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
assert "Báo Cáo Sàng Lọc Nguy Cơ Sức Khỏe" in res.get_data(as_text=True)

# 5. Kiểm thử POST /predict với dữ liệu lỗi (Glucose âm hoặc vượt ngưỡng)
res = client.post('/predict', data={'glucose': '-10', 'bmi': '25', 'age': '30'})
print(f"5. POST /predict (Lỗi validate): Status {res.status_code} - {'PASS' if 'Lỗi nhập liệu' in res.get_data(as_text=True) else 'FAIL'}")
assert "Lỗi nhập liệu" in res.get_data(as_text=True)

# 6. Kiểm thử Trang Giới thiệu mô hình GET /about
res = client.get('/about')
print(f"6. GET /about: Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
assert "Bảng So Sánh Hiệu Năng 3 Thuật Toán" in res.get_data(as_text=True)

# 7. Kiểm thử RESTful API POST /api/predict (JSON)
api_payload = {'glucose': 140.0, 'bmi': 29.5, 'age': 42}
res = client.post('/api/predict', json=api_payload)
print(f"7. POST /api/predict (JSON chuẩn): Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
data = res.get_json()
assert data['status'] == 'success'
assert 'prediction' in data
assert 'probabilities' in data
print(f"   -> Kết quả API: Nhãn = {data['prediction']} ({data['label']}) | Confidence = {data['confidence_score_pct']}%")

# 8. Kiểm thử RESTful API GET /api/metrics
res = client.get('/api/metrics')
print(f"8. GET /api/metrics: Status {res.status_code} - {'PASS' if res.status_code == 200 else 'FAIL'}")
assert res.status_code == 200
metrics_res = res.get_json()
assert metrics_res['status'] == 'success'
print(f"   -> Mô hình tốt nhất ghi nhận: {metrics_res['data'].get('best_model')}")

print("\n" + "=" * 60)
print("TẤT CẢ 8 BÀI KIỂM THỬ FLASK ĐỀU ĐẠT CHUẨN 100%!")
print("=" * 60)

