# NỘI DUNG SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN & BỘ CÂU HỎI VẤN ĐÁP (Q&A)

> **Môn học:** Học Máy (Machine Learning)  
> **Đề tài:** Xây dựng mô hình học máy hỗ trợ dự đoán tình trạng tiểu đường dựa trên các chỉ số y tế cơ bản (BMI, Tuổi, Lượng đường trong máu)  
> **Thời lượng thuyết trình chuẩn:** 10 - 15 phút  

---

# PHẦN 1: CẤU TRÚC 12 SLIDE BẢO VỆ ĐỒ ÁN

---

### SLIDE 1: TRANG TIÊU ĐỀ
- **Tên đề tài:** XÂY DỰNG MÔ HÌNH HỌC MÁY HỖ TRỢ DỰ ĐOÁN TÌNH TRẠNG TIỂU ĐƯỜNG DỰA TRÊN CÁC CHỈ SỐ Y TẾ CƠ BẢN
- **Sản phẩm triển khai:** Ứng dụng Web Sàng Lọc Sức Khỏe & RESTful API (DiaPredict ML)
- **Sinh viên thực hiện:** Nhóm sinh viên Công nghệ Thông tin
- **Giảng viên hướng dẫn:** Bộ môn Khoa học Dữ liệu / Trí tuệ Nhân tạo
- **Thời gian:** Năm 2026

---

### SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU ĐỒ ÁN
- **Bối cảnh y tế:**
  - Bệnh đái tháo đường đang gia tăng với tốc độ báo động toàn cầu (> 537 triệu người mắc).
  - Giai đoạn tiền tiểu đường không có triệu chứng rõ rệt, gần 50% người bệnh không biết mình mắc bệnh.
- **Vấn đề đặt ra:**
  - Làm thế nào để xây dựng một công cụ sàng lọc trực tuyến, nhanh chóng, chi phí 0 đồng, ai cũng có thể tự kiểm tra?
- **Mục tiêu của đề tài:**
  - Chỉ sử dụng **3 chỉ số lâm sàng cơ bản, dễ tiếp cận nhất**: Glucose, BMI, Age.
  - Xây dựng mô hình Machine Learning chuẩn mực, phòng chống rò rỉ dữ liệu (Data Leakage).
  - Triển khai sản phẩm Website thực tế có thể ứng dụng ngay.

---

### SLIDE 3: BỘ DỮ LIỆU & 3 CHỈ SỐ LÂM SÀNG CỐT LÕI
- **Nguồn dữ liệu:** Pima Indians Diabetes Dataset (Kaggle / UCI Machine Learning Repository).
  - Quy mô: 768 bệnh nhân.
  - Thuộc tính mục tiêu `Outcome`: 0 (Không tiểu đường - 65.1%) và 1 (Tiểu đường - 34.9%).
- **3 Chỉ số được chọn lọc:**
  1. **Glucose (mg/dL):** Đường huyết huyết tương. Yếu tố phản ánh trực tiếp khả năng chuyển hóa đường ($r \approx 0.49$).
  2. **BMI ($kg/m^2$):** Chỉ số khối cơ thể. Thừa cân, béo phì là nguyên nhân cốt lõi gây kháng insulin ($r \approx 0.31$).
  3. **Age (Năm):** Độ tuổi. Tuổi tác làm suy giảm tự nhiên chức năng của tế bào beta đảo tụy ($r \approx 0.24$).

---

### SLIDE 4: KHÁM PHÁ DỮ LIỆU (EDA) & PHÁT HIỆN BẤT THƯỜNG
- **Dữ liệu thiếu ẩn (Hidden Missing Values):**
  - Không có trường `NaN` trong file CSV, nhưng phát hiện:
    - `Glucose = 0`: Có **5 mẫu** (0.65%).
    - `BMI = 0`: Có **11 mẫu** (1.43%).
  - *Ý nghĩa y khoa:* Người sống không thể có đường huyết hoặc BMI bằng 0. Đây là lỗi kỹ thuật ghi chép khi dữ liệu bị thiếu.
- **Biện pháp xử lý:**
  - Thay thế số 0 bằng `NaN`.
  - Sử dụng phương pháp điền khuyết bằng **Trung vị (Median)** thay vì Trung bình (Mean) do dữ liệu có chứa điểm ngoại lai (Outliers).

---

### SLIDE 5: CHIẾN LƯỢC CHỐNG RÒ RỈ DỮ LIỆU (NO DATA LEAKAGE)
- **Rủi ro Data Leakage:** Nếu tính toán thống kê (Median, Mean, Std) trên toàn bộ dữ liệu trước khi chia train/test, mô hình sẽ bị rò rỉ phân bố tập kiểm thử, dẫn đến đánh giá thiếu trung thực.
- **Giải pháp của nhóm:**
  - **Bước 1:** Phân chia tập Train (80% - 614 mẫu) và Test (20% - 154 mẫu) với tham số `stratify=y` để giữ nguyên tỷ lệ nhãn.
  - **Bước 2:** Đóng gói quy trình thành `Pipeline`:
    - `SimpleImputer(strategy='median')`: Chỉ `.fit()` trên tập Train.
    - `StandardScaler()`: Chỉ `.fit()` trên tập Train.
    - `Classifier`: Huấn luyện thuật toán trên tập Train.
  - **Bước 3:** Áp dụng `.transform()` sang tập Test $\rightarrow$ Đảm bảo tính khách quan 100%.

---

### SLIDE 6: KẾT QUẢ THỰC NGHIỆM SO SÁNH 3 THUẬT TOÁN
*Toàn bộ số liệu lấy trực tiếp từ quá trình đo đạc thực tế trên 154 mẫu tập Test:*

| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 70.78% | 59.57% | 51.85% | 55.45% | 78.63% |
| **Decision Tree (Tối ưu)** | **79.87%** | **70.91%** | **72.22%** | **71.56%** | **81.44%** |
| **Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | **81.46%** |

- **Nhận xét sơ bộ:** Cả 3 mô hình đều hội tụ tốt. Decision Tree và Random Forest có ROC-AUC vượt 81.4%.

---

### SLIDE 7: PHÂN TÍCH MA TRẬN NHẦM LẪN & LÝ DO CHỌN DECISION TREE
- **Tiêu chuẩn Y tế:** Hạn chế tối đa ca **False Negative (Âm tính giả - Bỏ sót người có bệnh)** $\rightarrow$ Ưu tiên chỉ số **Recall (Độ nhạy)**.
- **So sánh Ma trận nhầm lẫn (154 mẫu Test gồm 54 ca bệnh):**
  - *Logistic Regression:* Bỏ sót **26 ca** bệnh (Recall = 51.85%).
  - *Random Forest:* Bỏ sót **21 ca** bệnh (Recall = 61.11%).
  - *Decision Tree:* Chỉ bỏ sót **15 ca** bệnh (Recall = **72.22%**).
- **Kết luận:** **Decision Tree** là mô hình tối ưu nhất cho bài toán sàng lọc này:
  - Accuracy cao nhất (79.87%).
  - Recall cao nhất (72.22%).
  - F1-Score cao nhất (71.56%).
  - Có tính minh bạch và giải thích được (Explainable AI) phù hợp cho y khoa.

---

### SLIDE 8: KIẾN TRÚC TRIỂN KHAI HỆ THỐNG WEB FLASK
- **Mô hình kiến trúc MVC mở rộng:**
  - **Tầng Client:** HTML5, CSS3 hiện đại, Vanilla JavaScript (không phụ thuộc framework nặng).
  - **Tầng Controller:** Flask Framework (Python), xử lý routing, validate dữ liệu 2 lớp.
  - **Tầng Model:** Nạp tệp `diabetes_model.pkl` một lần duy nhất vào bộ nhớ RAM khi server khởi động.
- **Đặc tính kỹ thuật:**
  - Không huấn luyện lại mô hình trong từng request $\rightarrow$ Tốc độ dự đoán tức thì (< 15ms).
  - Cung cấp song song 2 cổng: Web tương tác và RESTful API (`POST /api/predict`).

---

### SLIDE 9: DEMO CÁC CHỨC NĂNG NỔI BẬT TRÊN GIAO DIỆN
- **1. Bộ tính nhanh BMI thông minh:** Cho phép người dùng nhập trực tiếp Chiều cao (cm) và Cân nặng (kg), bấm "Tính & Điền" để tự động tính BMI theo chuẩn WHO Châu Á.
- **2. Form Validation 2 lớp:** Ngăn chặn tuyệt đối số âm, ký tự chữ hoặc các giá trị vượt ngưỡng sinh lý người.
- **3. Thước đo xác suất (Probability Gauge):** Hiển thị rõ ràng phần trăm tin cậy (ví dụ: 86.6% nguy cơ có bệnh).
- **4. Phân tích lâm sàng & Khuyến nghị:** Đánh giá từng chỉ số (Bình thường / Cảnh báo / Nguy cơ cao), hướng dẫn chế độ ăn uống, tập luyện và nhắc nhở xét nghiệm chuyên sâu.
- **5. Nút in phiếu kết quả:** Tiện ích in ấn nhanh cho người dùng.

---

### SLIDE 10: KIỂM THỬ HỆ THỐNG & ĐÁNH GIÁ NGHIỆM THU
- **Bộ kiểm thử tự động (`test_app.py`):**
  - Kiểm thử 8 kịch bản bao gồm: Trang chủ, Form nhập liệu, Phân loại ca nguy cơ cao, Phân loại ca an toàn, Bắt lỗi validate boundary, Trang báo cáo, API JSON và API Metrics.
  - **Kết quả:** Đạt chuẩn **100% (8/8 Test PASS)**.
- **Khả năng phòng vệ:**
  - Hệ thống tự động điền trung vị nếu người dùng chưa kịp đo đường huyết (nhập 0), không gây sập ứng dụng (Crash-free).

---

### SLIDE 11: ĐÓNG GÓP CỦA ĐỀ TÀI & BÀI HỌC KINH NGHIỆM
- **Đóng góp của đề tài:**
  - Hoàn thành đầy đủ 10 sản phẩm theo đúng yêu cầu đề bài.
  - Xây dựng được một sản phẩm ứng dụng hoàn chỉnh, có tính thẩm mỹ và giá trị thực tế cao.
  - Tuân thủ chuẩn mực chống Data Leakage và minh bạch số liệu đo đạc.
- **Bài học kinh nghiệm:**
  - Tầm quan trọng của việc hiểu bản chất bài toán miền ứng dụng (Domain Knowledge) trong y tế để chọn đúng độ đo đánh giá (Recall thay vì chỉ nhìn vào Accuracy).
  - Sức mạnh của Scikit-learn Pipeline trong việc đảm bảo tính nhất quán từ lúc huấn luyện đến lúc triển khai web.

---

### SLIDE 12: LỜI CẢM ƠN & PHIÊN HỎI ĐÁP (Q&A)
- **Tuyên bố:** Sản phẩm phục vụ mục đích nghiên cứu học tập, luôn đi kèm cảnh báo miễn trừ trách nhiệm y khoa.
- Xin chân thành cảm ơn Quý Thầy/Cô trong Hội đồng chấm thi đã lắng nghe!
- Nhóm chúng em rất mong nhận được những góp ý quý báu của Thầy/Cô.

---

# PHẦN 2: BỘ 10 CÂU HỎI VẤN ĐÁP (Q&A) VÀ CÂU TRẢ LỜI MẪU XUẤT SẮC

---

#### CÂU 1: Tại sao nhóm chỉ sử dụng 3 đặc trưng (Glucose, BMI, Age) thay vì dùng toàn bộ 8 đặc trưng gốc trong bộ dữ liệu Pima?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, việc chọn 3 đặc trưng xuất phát từ 2 lý do chính:  
> 1. **Về mặt tiếp cận người dùng (Usability):** Các chỉ số như Insulin huyết thanh sau 2 giờ hay Độ dày nếp gấp da cơ tam đầu (SkinThickness) là các xét nghiệm phức tạp mà người dân bình thường không thể biết nếu không có hồ sơ bệnh án chuyên sâu. Trong khi đó, Glucose (đường huyết), BMI (tính từ chiều cao/cân nặng) và Tuổi là những chỉ số phổ biến nhất mà ai cũng có thể tự cung cấp ngay tại nhà.  
> 2. **Về mặt khoa học dữ liệu (Feature Importance):** Qua bước phân tích tương quan Pearson (EDA), Glucose ($r=0.49$), BMI ($r=0.31$) và Age ($r=0.24$) chính là 3 thuộc tính có tương quan mạnh nhất và ổn định nhất với biến mục tiêu Outcome. Vì vậy, việc tinh gọn về 3 đặc trưng giúp mô hình vừa dễ ứng dụng trong thực tế mà vẫn giữ được độ chính xác xấp xỉ 80%."

---

#### CÂU 2: Hiện tượng Rò rỉ dữ liệu (Data Leakage) là gì? Nhóm đã phòng chống nó như thế nào trong mã nguồn?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, Data Leakage xảy ra khi thông tin từ tập kiểm thử (Test) vô tình bị rò rỉ vào quá trình huấn luyện mô hình, khiến kết quả đánh giá bị thổi phồng giả tạo (Over-optimistic).  
> Trong đề tài này, nhóm đã phòng chống triệt để qua 3 bước:  
> 1. **Tách dữ liệu trước:** Nhóm chia Train (80%) và Test (20%) bằng `train_test_split(stratify=y)` ngay từ đầu, trước khi thực hiện bất kỳ phép tính toán thống kê nào.  
> 2. **Chỉ `fit` trên Train:** Bộ điền khuyết `SimpleImputer` chỉ học giá trị trung vị từ tập Train (Glucose=117, BMI=32.4). Bộ chuẩn hóa `StandardScaler` chỉ học Mean và Std từ tập Train.  
> 3. **Đóng gói Pipeline:** Nhóm đóng gói cả Imputer, Scaler và Mô hình vào một đối tượng `Pipeline` duy nhất. Tập Test chỉ được gọi hàm `transform` mà tuyệt đối không tham gia vào bước `fit`. Bằng chứng thực nghiệm là Mean của tập Test sau chuẩn hóa là `[-0.0025, 0.0062, -0.053]`, khác 0, chứng minh tập Test hoàn toàn độc lập."

---

#### CÂU 3: Tại sao lại chuyển các giá trị 0 ở Glucose và BMI thành NaN và điền khuyết bằng Trung vị (Median) thay vì Trung bình (Mean)?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô:  
> - **Lý do chuyển 0 thành NaN:** Về mặt sinh học người, lượng đường huyết bằng 0 mg/dL hoặc BMI bằng 0 là bất khả thi ở người sống. Trong bộ dữ liệu Pima, giá trị 0 ở hai cột này thực chất là dữ liệu bị thiếu trong quá trình ghi chép. Nếu để nguyên số 0, các thuật toán (như Logistic Regression hay Scaler) sẽ coi số 0 là một giá trị toán học thực, làm sai lệch toàn bộ tâm phân bố dữ liệu.  
> - **Lý do chọn Trung vị (Median):** Khi vẽ biểu đồ Boxplot trong phần EDA, nhóm phát hiện thuộc tính BMI và Age có xuất hiện các điểm ngoại lai (Outliers) ở phía giá trị cao (BMI $> 50$, Age $> 65$). Trung bình cộng (Mean) rất nhạy cảm và dễ bị kéo lệch bởi các điểm ngoại lai này, trong khi Trung vị (Median) có tính chất bền vững (Robust) vượt trội, đại diện chính xác hơn cho trung tâm phân bố của đa số người bệnh."

---

#### CÂU 4: Tại sao Decision Tree lại đạt kết quả tốt hơn Random Forest trong đề tài này, mặc dù Random Forest là mô hình Ensemble phức tạp hơn?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, đây là một điểm thực nghiệm rất thú vị của đề tài:  
> 1. **Số chiều không gian đặc trưng nhỏ (Feature Space):** Random Forest phát huy sức mạnh tối đa khi dữ liệu có nhiều chiều (hàng chục đến hàng trăm thuộc tính), nhờ cơ chế chọn ngẫu nhiên tập con các đặc trưng (Feature Subsampling) tại mỗi nút để làm đa dạng các cây. Tuy nhiên, bài toán này chỉ có đúng 3 đặc trưng. Khi số đặc trưng quá ít, các cây trong rừng trở nên tương đối giống nhau, làm giảm ưu thế phân tán phương sai của Random Forest.  
> 2. **Cơ chế biểu quyết đa số (Majority Voting):** Với tập dữ liệu có tỷ lệ nhãn chênh lệch (65% : 35%), cơ chế biểu quyết bình quân của 100 cây khiến Random Forest có xu hướng thận trọng hơn ở nhãn 1 (Dương tính), dẫn đến Recall bị kéo xuống còn 61.11%. Trong khi đó, một cây Decision Tree đơn lẻ đã được tinh chỉnh độ sâu `max_depth=4` và `min_samples_leaf=5` bắt trọn các phân vùng y tế một cách sắc nét hơn, đạt Recall cao hơn hẳn (**72.22%** so với 61.11%) và F1-Score cao hơn (**71.56%** so with 64.08%)."

---

#### CÂU 5: Trong bài toán y tế này, giữa Precision và Recall thì chỉ số nào quan trọng hơn và tại sao?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, trong bài toán sàng lọc y tế dự phòng, **Recall (Độ nhạy)** quan trọng hơn Precision vì:  
> - **Hậu quả của False Negative (bỏ sót bệnh):** Một người thực sự đang bị tiền tiểu đường hoặc tiểu đường nhưng mô hình lại kết luận là 'Nguy cơ thấp (Âm tính)'. Người này sẽ chủ quan, không đi khám, tiếp tục lối sống có hại và đối mặt với nguy cơ biến chứng suy thận hoặc đột quỵ. Đây là rủi ro y khoa có thể ảnh hưởng đến tính mạng.  
> - **Hậu quả của False Positive (báo nhầm):** Một người khỏe mạnh bị mô hình cảnh báo 'Có nguy cơ'. Họ sẽ đến cơ sở y tế làm xét nghiệm máu kiểm tra lại. Bác sĩ xác nhận họ bình thường, không gây nguy hiểm tính mạng.  
> Do đó, mô hình có Recall càng cao (tức là FN càng nhỏ) thì càng có giá trị trong việc bảo vệ sức khỏe cộng đồng. Nhóm đã chọn Decision Tree chính vì mô hình này có Recall cao nhất (72.22%)."

---

#### CÂU 6: Nhóm đã làm gì để kiểm soát hiện tượng Overfitting ở cây quyết định (Decision Tree)?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, nếu để Decision Tree phát triển tự nhiên (không giới hạn), cây sẽ phân nhánh cho đến khi mọi nút lá đều thuần khiết (pure), dẫn đến học thuộc lòng cả các điểm nhiễu của tập Train (Overfitting).  
> Nhóm đã áp dụng kỹ thuật tỉa nhánh sớm (Pre-pruning) thông qua 3 siêu tham số quan trọng:  
> 1. `max_depth=4`: Giới hạn độ sâu tối đa của cây là 4 tầng.  
> 2. `min_samples_split=10`: Một nút phải có ít nhất 10 mẫu dữ liệu thì mới được phép tiếp tục phân nhánh.  
> 3. `min_samples_leaf=5`: Mỗi nút lá cuối cùng phải chứa tối thiểu 5 mẫu.  
> Nhờ các ràng buộc này, mô hình duy trì được khả năng tổng quát hóa rất tốt trên tập kiểm thử (đạt 79.87% Accuracy)."

---

#### CÂU 7: Làm thế nào để đảm bảo rằng khi người dùng nhập dữ liệu trên Website, dữ liệu đó được xử lý hoàn toàn nhất quán với lúc huấn luyện?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, nhóm giải quyết vấn đề này bằng cách sử dụng **Scikit-learn Pipeline** kết hợp với thư viện **Joblib**:  
> - Khi huấn luyện trong `train_model.py`, cả 3 bước: `imputer` (điền khuyết) $\rightarrow$ `scaler` (chuẩn hóa z-score) $\rightarrow$ `classifier` (thuật toán phân loại) được gộp chung thành một đối tượng Pipeline duy nhất.  
> - Khi lưu bằng `joblib.dump(best_pipeline, 'diabetes_model.pkl')`, toàn bộ các giá trị trung vị, mean, std đã học được lưu kèm theo mô hình.  
> - Trong Backend Flask (`app.py`), khi người dùng gửi 3 chỉ số lên, server chỉ cần truyền mảng dữ liệu thô vào `pipeline.predict()` hoặc `pipeline.predict_proba()`. Đối tượng Pipeline sẽ tự động thực hiện điền khuyết và chuẩn hóa đúng theo các tham số của tập huấn luyện mà không cần viết lại bất kỳ dòng code chuẩn hóa thủ công nào, loại bỏ hoàn toàn nguy cơ lệch thang đo (Scale Mismatch)."

---

#### CÂU 8: Tại sao nhóm lại sử dụng tham số `stratify=y` khi chia tập dữ liệu thành Train và Test?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, bộ dữ liệu Pima có sự mất cân bằng nhãn: Lớp 0 chiếm 65.1% và Lớp 1 chiếm 34.9%.  
> Nếu chia ngẫu nhiên thuần túy mà không dùng `stratify=y`, có nguy cơ tập Test vô tình nhận quá ít mẫu lớp 1 (ví dụ chỉ 15-20%) hoặc quá nhiều mẫu lớp 1, dẫn đến kết quả đánh giá độ chính xác bị sai lệch so với thực tế.  
> Tham số `stratify=y` bắt buộc hàm `train_test_split` phải bảo toàn tỷ lệ nhãn 65:35 đồng đều trên cả tập Train (65.15% : 34.85%) và tập Test (64.94% : 35.06%), giúp việc đánh giá hiệu năng mô hình mang tính đại diện và khách quan nhất."

---

#### CÂU 9: Giả sử một người dùng chưa từng xét nghiệm máu và nhập chỉ số Glucose là 0 thì hệ thống xử lý như thế nào?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, đây là một kịch bản thực tế đã được nhóm tính toán và kiểm thử:  
> - Ở phía Backend Flask, hàm `validate_and_parse_input` sẽ tiếp nhận giá trị 0 và chuyển thành `np.nan`.  
> - Khi đưa vào Pipeline, bước `SimpleImputer` sẽ tự động thay thế giá trị thiếu này bằng giá trị trung vị của dân số mẫu đã học từ tập Train (Glucose = 117.0 mg/dL).  
> - Sau đó mô hình vẫn tiến hành tính toán bình thường dựa trên BMI, Tuổi và giá trị Glucose ước lượng, trả về kết quả kèm theo ghi chú phân tích rõ ràng: *'Chỉ số Glucose chưa đo, hệ thống đã tự động ước lượng theo trung vị chuẩn'*. Nhờ vậy, ứng dụng hoạt động rất ổn định, thân thiện và không bao giờ bị dừng đột ngột."

---

#### CÂU 10: Nếu có thêm thời gian và điều kiện phát triển tiếp đồ án trong tương lai, nhóm sẽ cải tiến những điểm gì?
> **Trả lời mẫu:**  
> "Thưa Thầy/Cô, nhóm hướng tới 3 hướng nâng cấp trọng tâm:  
> 1. **Mở rộng dữ liệu bản địa:** Thu thập thêm dữ liệu từ các bệnh viện hoặc viện nghiên cứu dinh dưỡng tại Việt Nam để huấn luyện lại mô hình phù hợp hơn với đặc điểm thể tạng và lối sống của người Việt.  
> 2. **Chế độ nhập liệu nâng cao (Advanced Mode):** Cho phép người dùng tùy chọn nhập thêm các chỉ số chuyên sâu nếu họ đã có kết quả xét nghiệm máu (như HbA1c, Huyết áp, Cholesterol, Tiền sử gia đình) để tăng độ chính xác lên trên 85-90%.  
> 3. **Tích hợp Trợ lý AI Đa phương thức:** Đóng gói thành ứng dụng di động hoặc Chatbot trên Zalo/Telegram, cho phép người dùng chụp ảnh phiếu xét nghiệm máu, sử dụng OCR (nhận dạng ký tự quang học) để tự động trích xuất chỉ số Glucose và BMI mà không cần người dùng tự gõ tay."

