# HỆ THỐNG HỖ TRỢ DỰ ĐOÁN NGUY CƠ TIỂU ĐƯỜNG (DIAPREDICT ML)
> **Đồ án môn học:** Học Máy (Machine Learning)  
> **Bộ dữ liệu:** Pima Indians Diabetes Dataset (UCI Machine Learning Repository / Kaggle)  
> **Công nghệ:** Python 3.14, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Flask, HTML5, CSS3, JavaScript, Joblib  

---

## 1. Giới thiệu đề tài
Tiểu đường (Đái tháo đường) là một trong những bệnh mạn tính không lây phổ biến nhất toàn cầu, gây ra nhiều biến chứng nghiêm trọng về tim mạch, suy thận và thị lực nếu không được phát hiện kịp thời.

Đồ án này xây dựng một giải pháp Machine Learning hoàn chỉnh hỗ trợ sàng lọc sớm nguy cơ mắc bệnh tiểu đường dựa trên **3 chỉ số lâm sàng cơ bản và phổ biến nhất**:
1. **Glucose (Lượng đường huyết trong huyết tương - mg/dL)**
2. **BMI (Chỉ số khối cơ thể - $kg/m^2$)**
3. **Age (Độ tuổi - Năm)**

Hệ thống cung cấp một Website y tế hiện đại cho phép người dùng tự kiểm tra sức khỏe, nhận nhãn dự đoán nguy cơ (0: Thấp / 1: Cao), xem thước đo xác suất phần trăm tin cậy của mô hình, đọc phân tích chi tiết và nhận lời khuyên y tế tham khảo.

> ⚠️ **LƯU Ý Y TẾ QUAN TRỌNG:**  
> Đây là sản phẩm học tập và nghiên cứu thuật toán, **tuyệt đối không thay thế chẩn đoán chuyên môn, xét nghiệm sinh hóa máu hoặc phác đồ điều trị của bác sĩ chuyên khoa**.

---

## 2. Cấu trúc thư mục đồ án
```text
diabetes_prediction/
│
├── app.py                     # Máy chủ Flask Backend & RESTful API
├── train_model.py             # Script huấn luyện, đánh giá & đóng gói Pipeline
├── eda_and_preprocess.py      # Script tự động phân tích dữ liệu & xuất biểu đồ
├── test_app.py                # Test Suite kiểm thử 8 kịch bản tự động
├── requirements.txt           # Danh mục thư viện phụ thuộc
├── README.md                  # Hướng dẫn chi tiết cài đặt và sử dụng
├── report.md                  # Báo cáo học thuật toàn diện đồ án
├── presentation.md            # Nội dung slide thuyết trình & Bộ câu hỏi vấn đáp Q&A
│
├── data/
│   └── diabetes.csv           # Dữ liệu gốc chuẩn Pima Indians Diabetes (768 dòng)
│
├── models/
│   ├── diabetes_model.pkl     # Pipeline hoàn chỉnh (Imputer + Scaler + DecisionTree)
│   └── metrics.json           # Dữ liệu đo đạc thực tế của 3 mô hình & siêu tham số
│
├── static/
│   ├── css/
│   │   └── style.css          # CSS giao diện y tế hiện đại (Blue & White, Responsive)
│   ├── js/
│   │   └── script.js          # JS validation form, tính nhanh BMI & hiệu ứng UX
│   └── images/                # 6 Biểu đồ trực quan hóa y tế chất lượng cao
│       ├── outcome_distribution.png
│       ├── feature_distributions.png
│       ├── correlation_matrix.png
│       ├── feature_boxplots.png
│       ├── model_comparison.png
│       └── confusion_matrix_all.png
│
├── templates/
│   ├── base.html              # Layout chung: Navbar, Banner cảnh báo y tế, Footer
│   ├── index.html             # Trang chủ: Giới thiệu, quy trình 3 bước, ý nghĩa chỉ số
│   ├── predict.html           # Trang nhập liệu: Form chuẩn hóa kèm Bộ tính nhanh BMI
│   ├── result.html            # Trang kết quả: Nhãn dự đoán, xác suất %, phân tích & in
│   └── about.html             # Trang báo cáo: Bảng so sánh 3 mô hình, biểu đồ & tài liệu API
│
└── notebooks/
    └── analysis.ipynb         # Jupyter Notebook phân tích EDA chi tiết từng bước
```

---

## 3. Cài đặt môi trường & Thư viện

### Bước 1: Chuẩn bị môi trường Python
Yêu cầu Python 3.9 trở lên (đã được kiểm thử tương thích hoàn hảo trên Python 3.14 Windows 64-bit).

```bash
# Di chuyển vào thư mục dự án
cd diabetes_prediction

# (Tùy chọn) Khởi tạo môi trường ảo venv
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source venv/bin/activate
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn thực thi từng giai đoạn

### 4.1. Phân tích dữ liệu thăm dò (EDA) & Xuất biểu đồ
Lệnh này sẽ phân tích 768 dòng dữ liệu, phát hiện giá trị 0 ẩn ở Glucose và BMI, tạo ra 4 biểu đồ phân tích lưu vào `static/images/`:
```bash
python eda_and_preprocess.py
```

### 4.2. Huấn luyện 3 mô hình & Đóng gói Pipeline tối ưu
Lệnh này sẽ huấn luyện 3 thuật toán (**Logistic Regression, Decision Tree, Random Forest**), đánh giá trên 154 mẫu tập Test (80/20 có `stratify=y`), tạo biểu đồ so sánh mô hình, ma trận nhầm lẫn và lưu Pipeline tốt nhất vào `models/diabetes_model.pkl`:
```bash
python train_model.py
```

### 4.3. Kiểm thử tự động hệ thống Flask
Chạy toàn bộ 8 bài test tích hợp (Trang chủ, Form, Phân loại 0/1, Lỗi validation, API JSON):
```bash
python test_app.py
```

### 4.4. Khởi chạy ứng dụng Web Flask
```bash
python app.py
```
Sau đó mở trình duyệt web bất kỳ và truy cập địa chỉ:  
👉 **`http://127.0.0.1:5000/`**

---

## 5. Kết quả thực nghiệm đo đạc thực tế (Test Set 154 mẫu)

| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 70.78% | 59.57% | 51.85% | 55.45% | 78.63% |
| **Decision Tree (Tối ưu)** | **79.87%** | **70.91%** | **72.22%** | **71.56%** | **81.44%** |
| **Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | **81.46%** |

> 🏆 **Quyết định lựa chọn mô hình:**  
> **Decision Tree** được lựa chọn làm mô hình triển khai chính thức bởi vì mô hình đạt **Recall cao nhất (72.22%)** và **F1-Score cao nhất (71.56%)**, giúp hạn chế tối đa việc bỏ sót người có nguy cơ mắc bệnh (chỉ 15 ca False Negative trên 54 ca dương tính).

---

## 6. Tài liệu RESTful API Backend

### API 1: Dự đoán nguy cơ tiểu đường
- **Endpoint:** `POST /api/predict`
- **Headers:** `Content-Type: application/json`
- **Request Body mẫu:**
  ```json
  {
    "glucose": 145.0,
    "bmi": 28.5,
    "age": 42
  }
  ```
- **Response mẫu (HTTP 200 OK):**
  ```json
  {
    "status": "success",
    "input": {
      "glucose": 145.0,
      "bmi": 28.5,
      "age": 42.0
    },
    "prediction": 1,
    "label": "Nguy cơ tiểu đường (Dương tính)",
    "probabilities": {
      "negative_class_0_pct": 27.5,
      "positive_class_1_pct": 72.5
    },
    "confidence_score_pct": 72.5,
    "interpretations": [...],
    "recommendations": [...],
    "disclaimer": "Kết quả từ hệ thống chỉ mang tính chất tham khảo học tập, không thay thế chẩn đoán y khoa từ bác sĩ."
  }
  ```

### API 2: Lấy dữ liệu đánh giá 3 mô hình
- **Endpoint:** `GET /api/metrics`
- **Response:** Trả về toàn bộ chỉ số đánh giá thực tế và siêu tham số của 3 mô hình từ `metrics.json`.

---

## 7. Các điểm nổi bật về mặt kỹ thuật
1. **Kiến trúc Pipeline khép kín:** Đóng gói `SimpleImputer(strategy='median')` + `StandardScaler()` + Thuật toán phân loại vào đối tượng Pipeline của Scikit-learn, giúp quá trình dự đoán trên Web hoàn toàn đồng bộ với môi trường huấn luyện.
2. **Phòng chống Data Leakage triệt để:** Tất cả các tham số thống kê (trung vị, mean, std) được học nghiêm ngặt **chỉ trên tập Train (80%)** sau khi chia bằng `train_test_split(stratify=y)`.
3. **Bộ tính nhanh BMI thông minh:** Cho phép người dùng nhập trực tiếp chiều cao và cân nặng để tự động tính toán và điền chỉ số BMI mà không cần tính toán thủ công.
4. **Hiệu năng cao:** Mô hình chỉ nạp 1 lần duy nhất vào bộ nhớ RAM khi khởi động server, thời gian phản hồi dự đoán dưới 15ms.

---

## 8. Tác giả & Thông tin đồ án
- **Đồ án môn học:** Học Máy (Machine Learning)
- **Sinh viên thực hiện:** Sinh viên Công nghệ Thông tin
- **Giảng viên hướng dẫn:** Bộ môn Khoa học Dữ liệu / Trí tuệ Nhân tạo
- **Năm thực hiện:** 2026
