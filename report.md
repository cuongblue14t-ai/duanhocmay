# BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY (MACHINE LEARNING)

---

**ĐỀ TÀI:**  
### XÂY DỰNG MÔ HÌNH HỌC MÁY HỖ TRỢ DỰ ĐOÁN TÌNH TRẠNG TIỂU ĐƯỜNG DỰA TRÊN CÁC CHỈ SỐ Y TẾ CƠ BẢN (BMI, TUỔI, LƯỢNG ĐƯỜNG TRONG MÁU)

- **Ngành đào tạo:** Công nghệ Thông tin  
- **Bộ môn:** Khoa học Dữ liệu & Trí tuệ Nhân tạo  
- **Sinh viên thực hiện:** Nhóm sinh viên CNTT  
- **Thời gian thực hiện:** Năm 2026  

---

## TÓM TẮT ĐỒ ÁN (ABSTRACT)

Đái tháo đường (tiểu đường) là một căn bệnh mạn tính nguy hiểm với tốc độ gia tăng nhanh chóng trên toàn cầu. Việc phát hiện sớm nguy cơ đóng vai trò quyết định trong việc ngăn chặn các biến chứng tim mạch, mù lòa và suy thận mãn tính. Tuy nhiên, việc thực hiện các gói xét nghiệm toàn diện thường tốn kém thời gian và chi phí đối với phần đông người dân. 

Đồ án này nghiên cứu và trực tiếp phát triển một hệ thống học máy hoàn chỉnh, có khả năng hỗ trợ sàng lọc nguy cơ tiểu đường một cách nhanh chóng chỉ dựa trên **3 chỉ số lâm sàng cơ bản, dễ tiếp cận nhất**: **Lượng đường huyết (Glucose)**, **Chỉ số khối cơ thể (BMI)** và **Độ tuổi (Age)** trích xuất từ bộ dữ liệu chuẩn **Pima Indians Diabetes Dataset**.

Quy trình nghiên cứu được triển khai nghiêm ngặt theo các chuẩn mực khoa học dữ liệu:
1. Phát hiện và xử lý các giá trị 0 phi thực tế về mặt sinh học ở Glucose và BMI bằng kỹ thuật gán `NaN` và điền khuyết bằng Trung vị (Median).
2. Thiết kế chiến lược phân chia dữ liệu $80/20$ có phân tầng (`stratify=y`) và đóng gói toàn bộ quy trình tiền xử lý vào `Pipeline` nhằm phòng chống triệt để hiện tượng rò rỉ dữ liệu (Data Leakage).
3. Huấn luyện, đo đạc và so sánh thực nghiệm 3 thuật toán học máy kinh điển: **Hồi quy Logistic (Logistic Regression)**, **Cây quyết định (Decision Tree)** và **Rừng ngẫu nhiên (Random Forest)**.
4. Kết quả thực nghiệm trên 154 mẫu kiểm thử độc lập cho thấy mô hình **Decision Tree** đạt hiệu năng vượt trội nhất với **Accuracy = 79.87%**, **Recall = 72.22%**, **F1-Score = 71.56%** và **ROC-AUC = 81.44%**, chứng minh tính tối ưu trong việc giảm thiểu tối đa các ca bỏ sót bệnh (False Negative).
5. Đóng gói mô hình bằng `Joblib` và triển khai một Website y tế hiện đại với Backend `Flask`, giao diện phong cách xanh dương - trắng trực quan, tích hợp công cụ tính nhanh BMI tương tác, thanh đo xác suất phần trăm tin cậy và hệ thống RESTful API phục vụ tích hợp.

---

## CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU ĐỀ TÀI

### 1.1. Tính cấp thiết của đề tài
Theo số liệu từ Hiệp hội Đái tháo đường Thế giới (IDF), toàn cầu hiện có hơn 537 triệu người trưởng thành mắc bệnh đái tháo đường, và gần một nửa trong số đó chưa từng được chẩn đoán. Tại Việt Nam, tỷ lệ người mắc đái tháo đường đang trẻ hóa nhanh chóng do thói quen ăn uống nhiều carbohydrate tinh chế, ít vận động và tình trạng thừa cân, béo phì.

Thách thức lớn nhất trong y tế cộng đồng là giai đoạn tiền đái tháo đường thường không có triệu chứng lâm sàng rõ rệt. Khi người bệnh bắt đầu cảm thấy mệt mỏi, sụt cân hay khát nước nhiều thì các tổn thương vi mạch đã bắt đầu xuất hiện. Do đó, việc ứng dụng Trí tuệ Nhân tạo và Machine Learning để xây dựng một công cụ sàng lọc trực tuyến, phi xâm lấn, chi phí 0 đồng giúp người dân tự đánh giá nguy cơ sức khỏe là một hướng tiếp cận có ý nghĩa xã hội và ứng dụng thực tiễn rất cao.

### 1.2. Mục tiêu nghiên cứu
- **Mục tiêu khoa học dữ liệu:** Khám phá quy luật phân bố của các chỉ số y tế, xử lý dữ liệu thiếu ẩn, xây dựng Pipeline tiền xử lý chuẩn mực không rò rỉ dữ liệu, huấn luyện và so sánh khách quan 3 thuật toán phân loại.
- **Mục tiêu kỹ thuật phần mềm:** Đóng gói mô hình Machine Learning thành công và tích hợp vào một ứng dụng Web hoàn chỉnh sử dụng Flask, có khả năng phản hồi tức thời dưới 15ms.
- **Mục tiêu người dùng:** Cung cấp trải nghiệm thân thiện, dễ hiểu, minh bạch về xác suất dự đoán và cung cấp các lời khuyên y tế tham khảo hữu ích, đi kèm cảnh báo miễn trừ trách nhiệm y khoa rõ ràng.

### 1.3. Phạm vi và Giới hạn đề tài
- **Phạm vi dữ liệu:** Sử dụng bộ dữ liệu Pima Indians Diabetes gồm 768 cá thể. Chỉ tập trung vào 3 đặc trưng cốt lõi: Glucose, BMI, Age.
- **Giới hạn kỹ thuật:** Tuân thủ yêu cầu đề tài không sử dụng Deep Learning (mạng nơ-ron sâu) nhằm duy trì tính minh bạch, khả năng giải thích (Explainable AI) và tốc độ tính toán nhanh của các mô hình học máy truyền thống.
- **Giới hạn y khoa:** Hệ thống đóng vai trò công cụ hỗ trợ sàng lọc tham khảo ban đầu, không thay thế xét nghiệm máu chuyên sâu hay chỉ định của bác sĩ y khoa.

---

## CHƯƠNG 2: TỔNG QUAN LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

### 2.1. Giới thiệu Bộ dữ liệu Pima Indians Diabetes
Bộ dữ liệu được thu thập ban đầu bởi Viện Quốc gia về Bệnh tiểu đường, Tiêu hóa và Bệnh thận (NIDDK) tại Hoa Kỳ, bao gồm các đối tượng nghiên cứu là phụ nữ người Mỹ bản địa gốc Pima từ 21 tuổi trở lên. Đây là quần thể có tỷ lệ mắc đái tháo đường typ 2 đặc biệt cao do sự biến đổi nhanh chóng về lối sống và chế độ dinh dưỡng.

Cấu trúc bộ dữ liệu gồm 768 dòng, mỗi dòng đại diện cho một bệnh nhân với nhãn mục tiêu `Outcome`:
- `Outcome = 0`: Âm tính (Không mắc tiểu đường).
- `Outcome = 1`: Dương tính (Có mắc tiểu đường).

### 2.2. Ý nghĩa lâm sàng của 3 đặc trưng được lựa chọn
1. **Glucose (mg/dL):** Nồng độ glucose trong huyết tương sau thử nghiệm dung nạp glucose đường uống 2 giờ. Đây là tiêu chuẩn vàng sinh hóa phản ánh trực tiếp khả năng chuyển hóa đường của cơ thể.
2. **BMI ($kg/m^2$):** Chỉ số khối cơ thể được tính bằng $BMI = \frac{\text{Cân nặng (kg)}}{[\text{Chiều cao (m)}]^2}$. Mỡ thừa trong cơ thể, đặc biệt là mỡ nội tạng, giải phóng các axit béo tự do và cytokine gây viêm, trực tiếp làm suy giảm độ nhạy của thụ thể insulin tại tế bào (hiện tượng kháng insulin).
3. **Age (Năm):** Quá trình lão hóa tự nhiên dẫn đến sự suy giảm dần về số lượng và chức năng của các tế bào beta đảo tụy chuyên trách tiết insulin.

### 2.3. Cơ sở lý thuyết của 3 thuật toán Machine Learning
1. **Hồi quy Logistic (Logistic Regression):**  
   Mô hình phân loại tuyến tính sử dụng hàm Sigmoid:
   $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{với } z = w_0 + w_1 \cdot \text{Glucose} + w_2 \cdot \text{BMI} + w_3 \cdot \text{Age}$$
   Đặc điểm: Đơn giản, tính toán nhanh, giải thích được trọng số, nhưng giả định mối quan hệ tuyến tính giữa log-odds và các biến độc lập.
2. **Cây quyết định (Decision Tree Classifier):**  
   Thuật toán phân chia không gian đặc trưng bằng cây nhị phân dựa trên tiêu chuẩn đo độ vẩn đục Gini Impurity:
   $$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$
   Đặc điểm: Bắt trọn các ranh giới phi tuyến tính rất tự nhiên, mô phỏng đúng cách các bác sĩ đưa ra quyết định sàng lọc qua từng ngưỡng chỉ số (ví dụ: Nếu Glucose $> 127.5$ và BMI $> 29.8$ thì xếp vào nhóm nguy cơ cao).
3. **Rừng ngẫu nhiên (Random Forest Classifier):**  
   Phương pháp học kết hợp (Ensemble Learning) áp dụng kỹ thuật Bagging (Bootstrap Aggregating). Thuật toán xây dựng một tập hợp gồm nhiều cây quyết định độc lập (trong đồ án này là 100 cây), mỗi cây được huấn luyện trên một tập mẫu con ngẫu nhiên có hoàn lại và chọn ngẫu nhiên tập con các đặc trưng tại mỗi điểm phân nhánh. Dự đoán cuối cùng được lấy theo biểu quyết đa số (Majority Voting).

### 2.4. Công nghệ Web & Backend
- **Flask Framework (Python):** Vi khung (microframework) nhẹ, linh hoạt, hiệu năng cao, cho phép kết nối trực tiếp với các thư viện khoa học dữ liệu Python mà không cần cầu nối trung gian.
- **Joblib:** Thư viện tối ưu hóa tuần tự hóa (serialization) các đối tượng Python chứa mảng số học lớn của NumPy/Scikit-learn, giúp lưu trữ và tải lại Pipeline mô hình trong vài mili-giây.
- **Frontend Hiện đại:** HTML5 ngữ nghĩa, CSS3 Flexbox/Grid responsive, JavaScript ES6 xử lý tương tác DOM không phụ thuộc thư viện nặng.

---

## CHƯƠNG 3: KHÁM PHÁ DỮ LIỆU (EDA) VÀ CHIẾN LƯỢC TIỀN XỬ LÝ

### 3.1. Thống kê mô tả và Dữ liệu thiếu ẩn (Hidden Missing Values)
Phân tích thực tế tệp dữ liệu `diabetes.csv` ghi nhận:
- Tổng số mẫu: 768 dòng, không có trường nào chứa giá trị `null` hoặc `NaN` theo định dạng text.
- Tuy nhiên, khi kiểm tra giá trị nhỏ nhất (`min`):
  - Cột `Glucose`: Giá trị nhỏ nhất là **0**, có **5 mẫu** mang giá trị 0 (0.65%).
  - Cột `BMI`: Giá trị nhỏ nhất là **0**, có **11 mẫu** mang giá trị 0 (1.43%).
  - Cột `Age`: Giá trị nhỏ nhất là 21, lớn nhất 81, không có giá trị 0.

*Đánh giá y khoa:* Một người sống không thể có lượng đường huyết bằng 0 hoặc thể trọng/chiều cao bằng 0. Đây là lỗi kỹ thuật ghi chép dữ liệu lâm sàng trong quá khứ khi nhân viên y tế gán số 0 cho các ca bệnh nhân chưa được làm xét nghiệm tương ứng.

*Giải pháp:* Chuyển đổi toàn bộ các số 0 ở hai cột này thành `np.nan` trước khi đưa vào bước tiền xử lý để tránh bóp méo phân bố thực tế.

### 3.2. Phân bố nhãn mục tiêu và Hiện tượng mất cân bằng lớp
Thống kê nhãn `Outcome`:
- Lớp 0 (Âm tính): 500 mẫu (**65.10%**).
- Lớp 1 (Dương tính): 268 mẫu (**34.90%**).

Tỷ lệ nhãn là khoảng $1.87 : 1$. Mặc dù chưa đến mức mất cân bằng cực đoan (Severe Imbalance), sự chênh lệch này đòi hỏi khi chia tập Train/Test theo tỷ lệ $80/20$, bắt buộc phải sử dụng tham số `stratify=y` trong hàm `train_test_split()`. Kết quả phân chia:
- Tập Train (80%): 614 mẫu (400 mẫu lớp 0, 214 mẫu lớp 1 - tỷ lệ 65.15% : 34.85%).
- Tập Test (20%): 154 mẫu (100 mẫu lớp 0, 54 mẫu lớp 1 - tỷ lệ 64.94% : 35.06%).

### 3.3. Ma trận tương quan và Phân tích tương quan Pearson
Tính toán hệ số tương quan $r$ giữa các biến:
- `Glucose` với `Outcome`: $r \approx 0.49$ (Tương quan dương mạnh nhất, là thuộc tính đóng vai trò quyết định hàng đầu).
- `BMI` với `Outcome`: $r \approx 0.31$ (Tương quan dương mức trung bình).
- `Age` với `Outcome`: $r \approx 0.24$ (Tương quan dương mức độ vừa).
- Mức tương quan giữa các đặc trưng đầu vào với nhau đều dưới 0.30, khẳng định không xảy ra hiện tượng đa cộng tuyến (Multicollinearity).

### 3.4. Chiến lược Tiền xử lý chống Rò rỉ Dữ liệu (Anti Data Leakage)
Rò rỉ dữ liệu (Data Leakage) xảy ra khi thông tin từ tập kiểm thử (Test) vô tình được đưa vào quá trình huấn luyện, khiến mô hình học được phân bố của tập kiểm thử trước khi được đánh giá.

*Quy trình chuẩn hóa được triển khai:*
1. Chia dữ liệu thành tập Train và Test trước tiên.
2. Khởi tạo `SimpleImputer(strategy='median')`: Sử dụng giá trị trung vị vì trung vị có tính bền vững (robust) cao trước các giá trị ngoại lai so với trung bình (mean). Bộ điền khuyết chỉ gọi hàm `.fit(X_train)`, học được:
   - Glucose trung vị: **117.00** mg/dL.
   - BMI trung vị: **32.40** $kg/m^2$.
   - Age trung vị: **29.00** tuổi.
3. Khởi tạo `StandardScaler()`: Chuẩn hóa thang đo theo công thức $z = \frac{x - \mu}{\sigma}$. Bộ chuẩn hóa chỉ gọi hàm `.fit()` trên dữ liệu Train đã điền khuyết, học được:
   - Glucose: $\mu = 121.67, \sigma = 29.98$.
   - BMI: $\mu = 32.45, \sigma = 6.82$.
   - Age: $\mu = 33.37, \sigma = 11.82$.
4. Áp dụng `.transform()` với các tham số đã đóng băng ở trên lên tập Test. Kết quả kiểm chứng: Tập Test sau chuẩn hóa có Mean là `[-0.0025, 0.0062, -0.0530]` (khác 0), chứng minh tính độc lập và khách quan tuyệt đối của tập Test.

---

## CHƯƠNG 4: HUẤN LUYỆN, THỰC NGHIỆM VÀ ĐÁNH GIÁ 3 MÔ HÌNH

### 4.1. Thiết lập siêu tham số cho các mô hình
Để đảm bảo so sánh công bằng và ngăn ngừa hiện tượng học vẹt (Overfitting), các mô hình được thiết lập siêu tham số như sau:
1. **Logistic Regression:** `random_state=42`, `max_iter=1000`, `C=1.0`, bộ giải tối ưu `solver='lbfgs'`.
2. **Decision Tree:** `random_state=42`, `max_depth=4` (khống chế độ sâu tối đa nhằm tránh chia nhánh quá mức vào các điểm nhiễu), `min_samples_split=10`, `min_samples_leaf=5`.
3. **Random Forest:** `random_state=42`, `n_estimators=100` (100 cây quyết định), `max_depth=5`, `min_samples_leaf=4`.

### 4.2. Bảng kết quả thực nghiệm đo đạc thực tế
Sau khi chạy script `train_model.py` trên 154 mẫu kiểm thử độc lập, các chỉ số thu được:

| Thuật toán | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | 70.78% | 59.57% | 51.85% | 55.45% | 78.63% |
| **2. Decision Tree (Tối ưu)** | **79.87%** | **70.91%** | **72.22%** | **71.56%** | **81.44%** |
| **3. Random Forest (100 cây)** | 75.97% | 67.35% | 61.11% | 64.08% | **81.46%** |

### 4.3. Phân tích chi tiết Ma trận nhầm lẫn (Confusion Matrix)
Ma trận nhầm lẫn đo đạc trên 154 mẫu tập Test (gồm 100 mẫu âm tính thực tế và 54 mẫu dương tính thực tế):

- **Logistic Regression:**
  - True Negative (TN) = 81 | False Positive (FP) = 19
  - False Negative (FN) = **26** | True Positive (TP) = 28
  - *Nhận xét:* Bỏ sót 26 trên tổng số 54 người có bệnh (FN cao), độ nhạy Recall chỉ đạt 51.85%. Ranh giới phân cách tuyến tính không đủ linh hoạt để phân biệt các trường hợp bệnh nhân ở vùng giao thoa chỉ số.

- **Decision Tree:**
  - True Negative (TN) = 84 | False Positive (FP) = 16
  - False Negative (FN) = **15** | True Positive (TP) = **39**
  - *Nhận xét:* Giảm đáng kể số ca bỏ sót bệnh từ 26 xuống còn 15. Phát hiện chính xác 39/54 ca có nguy cơ tiểu đường, đưa Recall lên mức cao nhất là **72.22%**, đồng thời giữ độ chính xác tổng quát cao nhất đạt **79.87%**.

- **Random Forest:**
  - True Negative (TN) = 84 | False Positive (FP) = 16
  - False Negative (FN) = 21 | True Positive (TP) = 33
  - *Nhận xét:* Random Forest cho điểm ROC-AUC rất tốt (81.46%) và khả năng tổng quát hóa cao. Tuy nhiên, trên không gian chỉ có 3 thuộc tính, cơ chế lấy mẫu ngẫu nhiên các đặc trưng và biểu quyết đa số của 100 cây khiến mô hình có xu hướng thận trọng hơn ở nhãn 1, dẫn đến Recall đạt 61.11% (thấp hơn Decision Tree đơn lẻ đã tinh chỉnh).

### 4.4. Quyết định lựa chọn mô hình triển khai
Trong bài toán ứng dụng y tế dự phòng, **Recall (Độ nhạy)** là tiêu chí được ưu tiên hàng đầu, bởi vì hậu quả của việc "bỏ sót một người thực sự có bệnh và kết luận họ bình thường" (False Negative) nguy hiểm hơn nhiều so với việc "cảnh báo một người khỏe mạnh đi kiểm tra lại" (False Positive).

Vì vậy, **Decision Tree** được lựa chọn làm mô hình triển khai chính thức cho hệ thống website vì:
1. Đạt **Accuracy cao nhất (79.87%)**.
2. Đạt **Recall cao nhất (72.22%)**.
3. Đạt **F1-Score cao nhất (71.56%)**.
4. Cây quyết định có tính giải thích trực quan cao, hoàn toàn tương thích với tư duy chẩn đoán phân tầng ngưỡng của y khoa.

Toàn bộ Pipeline gồm `SimpleImputer` + `StandardScaler` + `DecisionTreeClassifier` được lưu vào `models/diabetes_model.pkl` bằng thư viện `Joblib`.

---

## CHƯƠNG 5: THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG WEBSITE Y TẾ & FLASK BACKEND

### 5.1. Thiết kế kiến trúc tổng thể
Hệ thống được tổ chức thành 3 tầng rõ rệt:
1. **Tầng Giao diện (Presentation Layer):**
   - Áp dụng phong cách thiết kế Y tế Hiện đại (Clinical Medical Aesthetic) với bảng màu chủ đạo gồm Xanh dương y tế (`#1d4ed8`), Xanh ngọc (`#0d9488`), Trắng (`#ffffff`) và Xám nhạt (`#f8fafc`).
   - Tối ưu hóa trải nghiệm trên mọi kích thước màn hình (Responsive Design) từ máy tính bàn, máy tính bảng đến điện thoại di động thông qua CSS Flexbox và CSS Grid.
2. **Tầng Điều khiển (Application & Controller Layer):**
   - Viết bằng framework Flask trong tệp `app.py`.
   - Nạp mô hình `diabetes_model.pkl` một lần duy nhất lúc server khởi động.
   - Tiếp nhận yêu cầu, lọc và chuẩn hóa dữ liệu, gọi Pipeline suy luận và đóng gói kết quả phản hồi.
3. **Tầng Dịch vụ Dữ liệu & API (RESTful API Layer):**
   - Cung cấp điểm cuối `/api/predict` (HTTP POST JSON) và `/api/metrics` (HTTP GET JSON).

### 5.2. Các chức năng trên giao diện Web
1. **Trang chủ (`index.html`):**  
   Giới thiệu đề tài, hiển thị các thẻ số liệu thống kê (Độ chính xác 79.87%, 3 chỉ số lâm sàng, 768 hồ sơ dữ liệu), quy trình 3 bước sử dụng và ý nghĩa y khoa của từng chỉ số.
2. **Trang dự đoán (`predict.html`):**  
   Form nhập 3 chỉ số lâm sàng kèm kiểm tra dữ liệu đầu vào chặt chẽ. Đặc biệt, tích hợp **Bộ tính nhanh BMI thông minh**: người dùng chỉ cần nhập Chiều cao (cm) và Cân nặng (kg), bấm "Tính & Điền", hệ thống JavaScript phía client sẽ tự động tính BMI và điền vào form.
3. **Trang kết quả (`result.html`):**  
   Hiển thị nhãn dự đoán (0: Nguy cơ thấp / 1: Cảnh báo nguy cơ cao), thanh tiến trình xác suất (Probability Gauge) thể hiện phần trăm tin cậy của mô hình, bảng so sánh từng chỉ số với ngưỡng y tế chuẩn, danh mục khuyến nghị sinh hoạt và nút in phiếu kết quả (`window.print()`).
4. **Trang thông tin mô hình (`about.html`):**  
   Trình bày tổng quan bộ dữ liệu, bảng so sánh hiệu năng 3 thuật toán nạp động từ `metrics.json`, nhúng 4 biểu đồ phân tích trực quan độ nét cao và tài liệu hướng dẫn gọi API.

---

## CHƯƠNG 6: KIỂM THỬ VÀ ĐÁNH GIÁ NGHIỆM THU HỆ THỐNG

### 6.1. Phương pháp kiểm thử
Xây dựng bộ kịch bản kiểm thử tự động trong tệp `test_app.py` sử dụng công cụ `test_client()` của Flask, kiểm tra đồng thời cả giao diện người dùng và API JSON.

### 6.2. Kết quả kiểm thử các kịch bản thực tế
1. **Kịch bản 1: Kiểm thử tải Trang chủ (`GET /`):** Mã trạng thái HTTP 200 OK, hiển thị đầy đủ thương hiệu DiaPredict ML và các nút chức năng.
2. **Kịch bản 2: Kiểm thử tải Form dự đoán (`GET /predict`):** Mã trạng thái HTTP 200 OK, form nhập liệu sẵn sàng tiếp nhận.
3. **Kịch bản 3: Kiểm thử dự đoán ca bệnh có nguy cơ cao:**  
   Input: Glucose = 165 mg/dL, BMI = 35.5 $kg/m^2$, Age = 50 tuổi.  
   Kết quả: Dự đoán chính xác Nhãn 1 (Dương tính), xác suất nguy cơ > 85%, hệ thống đưa ra khuyến nghị kiểm tra đường huyết tĩnh mạch.
4. **Kịch bản 4: Kiểm thử dự đoán ca người khỏe mạnh:**  
   Input: Glucose = 85 mg/dL, BMI = 21.0 $kg/m^2$, Age = 24 tuổi.  
   Kết quả: Dự đoán chính xác Nhãn 0 (Âm tính), xác suất an toàn > 98%.
5. **Kịch bản 5: Kiểm thử phòng vệ dữ liệu lỗi (Boundary & Negative Testing):**  
   Input: Glucose = -10, BMI = 150, Age = "abc".  
   Kết quả: Hệ thống chặn thành công, hiển thị thông báo lỗi tiếng Việt thân thiện, không làm ứng dụng bị gián đoạn hay crash server.
6. **Kịch bản 6: Kiểm thử tải Trang báo cáo (`GET /about`):** Mã trạng thái HTTP 200 OK, bảng so sánh và các biểu đồ tải hoàn hảo.
7. **Kịch bản 7: Kiểm thử RESTful API (`POST /api/predict`):** Nhận JSON, trả về JSON chuẩn RESTful kèm nhãn, xác suất và khuyến nghị trong vòng 12ms.
8. **Kịch bản 8: Kiểm thử API Metrics (`GET /api/metrics`):** Trả về đầy đủ số liệu đo đạc của 3 mô hình từ tệp `metrics.json`.

*Đánh giá nghiệm thu:* **100% các kịch bản kiểm thử đều đạt chuẩn (PASS)**.

---

## CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Các kết quả đã đạt được
1. Đã hoàn thành một đồ án Machine Learning hoàn chỉnh, bài bản từ khâu tiền xử lý, nghiên cứu thuật toán đến sản phẩm phần mềm thực tế.
2. Giải quyết triệt để vấn đề dữ liệu thiếu ẩn và kiểm soát chặt chẽ nguy cơ rò rỉ dữ liệu (No Data Leakage) bằng kiến trúc Scikit-learn Pipeline.
3. So sánh khách quan 3 thuật toán kinh điển với số liệu đo đạc thực tế 100%, chứng minh tính phù hợp của Decision Tree trong bài toán sàng lọc y tế (Accuracy 79.87%, Recall 72.22%).
4. Xây dựng website y tế hiện đại, chuyên nghiệp, responsive, dễ sử dụng kèm API RESTful hoàn chỉnh.

### 7.2. Hạn chế còn tồn tại
1. Bộ dữ liệu nghiên cứu có quy mô 768 mẫu, được thu thập trên một nhóm dân tộc cụ thể (người bản địa Pima), do đó cần thêm dữ liệu mở rộng trên thể tạng người Việt Nam để nâng cao hơn nữa độ chuẩn xác cục bộ.
2. Số lượng đặc trưng đầu vào được khống chế ở 3 chỉ số tối giản. Mặc dù tiện lợi cho người dùng, việc thiếu vắng các chỉ số như tiền sử gia đình (Pedigree Function) hay huyết áp có thể làm giới hạn khả năng phát hiện ở những ca bệnh cảnh đặc thù.

### 7.3. Hướng phát triển trong tương lai
1. Mở rộng thêm tính năng cho phép người dùng nhập tùy chọn các chỉ số xét nghiệm chuyên sâu (nếu có) như HbA1c, Huyết áp (Blood Pressure) hoặc Insulin để nâng cao độ chính xác lên trên 85%.
2. Tích hợp mô hình vào ứng dụng di động (Flutter / React Native) hoặc chatbot trợ lý sức khỏe tương tác qua Zalo/Telegram.
3. Kết nối dữ liệu theo dõi theo chuỗi thời gian (Time-series) để người dùng có thể theo dõi biểu đồ biến thiên đường huyết qua các tháng.

---

## TÀI LIỆU THAM KHẢO

1. **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).** *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* In Proceedings of the Symposium on Computer Applications and Medical Care (p. 261). American Medical Informatics Association.
2. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
3. **American Diabetes Association (ADA). (2024).** *Standards of Care in Diabetes—2024.* Diabetes Care, 47(Suppl. 1), S1–S343.
4. **Grinberg, M. (2018).** *Flask Web Development: Developing Web Applications with Python (2nd ed.).* O'Reilly Media.
5. **Kaggle Dataset Repository:** *Pima Indians Diabetes Database.* UCI Machine Learning Repository.

