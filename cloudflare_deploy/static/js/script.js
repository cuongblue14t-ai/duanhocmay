/**
 * JavaScript cho Hệ thống Dự đoán Nguy cơ Tiểu đường (DiaPredict ML)
 * Chức năng:
 *   - Kiểm tra dữ liệu đầu vào phía Client (Form Validation).
 *   - Bộ tính nhanh BMI từ chiều cao & cân nặng.
 *   - Hiệu ứng tải (Loading spinner) khi gửi yêu cầu.
 *   - Hiệu ứng thanh tiến trình xác suất.
 */

document.addEventListener('DOMContentLoaded', function () {
    
    // 1. XỬ LÝ FORM VALIDATION TRÊN TRANG DỰ ĐOÁN
    const predictionForm = document.getElementById('predictionForm');
    const glucoseInput = document.getElementById('glucose');
    const bmiInput = document.getElementById('bmi');
    const ageInput = document.getElementById('age');
    const submitBtn = document.getElementById('submitBtn');

    if (predictionForm) {
        // Xóa thông báo lỗi khi người dùng bắt đầu nhập lại
        [glucoseInput, bmiInput, ageInput].forEach(function (input) {
            input.addEventListener('input', function () {
                this.classList.remove('is-invalid');
                const errDiv = document.getElementById(this.id + 'Error');
                if (errDiv) errDiv.textContent = '';
            });
        });

        // Xử lý sự kiện Submit Form
        predictionForm.addEventListener('submit', function (e) {
            let isValid = true;

            // Kiểm tra Glucose (0 - 500)
            const glucoseVal = parseFloat(glucoseInput.value);
            if (isNaN(glucoseVal) || glucoseInput.value.trim() === '') {
                showError(glucoseInput, 'glucoseError', 'Vui lòng nhập chỉ số đường huyết.');
                isValid = false;
            } else if (glucoseVal < 0 || glucoseVal > 500) {
                showError(glucoseInput, 'glucoseError', 'Chỉ số Glucose phải từ 0 đến 500 mg/dL.');
                isValid = false;
            }

            // Kiểm tra BMI (0 - 100)
            const bmiVal = parseFloat(bmiInput.value);
            if (isNaN(bmiVal) || bmiInput.value.trim() === '') {
                showError(bmiInput, 'bmiError', 'Vui lòng nhập chỉ số khối cơ thể (BMI).');
                isValid = false;
            } else if (bmiVal < 0 || bmiVal > 100) {
                showError(bmiInput, 'bmiError', 'Chỉ số BMI phải từ 0 đến 100 kg/m².');
                isValid = false;
            }

            // Kiểm tra Tuổi (1 - 125)
            const ageVal = parseInt(ageInput.value, 10);
            if (isNaN(ageVal) || ageInput.value.trim() === '') {
                showError(ageInput, 'ageError', 'Vui lòng nhập độ tuổi.');
                isValid = false;
            } else if (ageVal < 1 || ageVal > 125) {
                showError(ageInput, 'ageError', 'Độ tuổi phải từ 1 đến 125 tuổi.');
                isValid = false;
            }

            if (!isValid) {
                e.preventDefault();
                return;
            }

            // Hiển thị hiệu ứng loading
            if (submitBtn) {
                submitBtn.disabled = true;
                const btnText = submitBtn.querySelector('.btn-text');
                const btnSpinner = submitBtn.querySelector('.btn-spinner');
                if (btnText) btnText.style.display = 'none';
                if (btnSpinner) btnSpinner.style.display = 'inline-block';
            }
        });
    }

    function showError(inputElement, errorElementId, message) {
        inputElement.classList.add('is-invalid');
        const errDiv = document.getElementById(errorElementId);
        if (errDiv) {
            errDiv.textContent = message;
        }
        inputElement.focus();
    }

    // 2. CÔNG CỤ TÍNH NHANH BMI (QUICK BMI CALCULATOR)
    const toggleBmiCalcBtn = document.getElementById('toggleBmiCalcBtn');
    const bmiQuickCalc = document.getElementById('bmiQuickCalc');
    const calcHeight = document.getElementById('calcHeight');
    const calcWeight = document.getElementById('calcWeight');
    const btnCalculateBmi = document.getElementById('btnCalculateBmi');
    const bmiCalcResult = document.getElementById('bmiCalcResult');

    if (toggleBmiCalcBtn && bmiQuickCalc) {
        toggleBmiCalcBtn.addEventListener('click', function () {
            if (bmiQuickCalc.style.display === 'none' || bmiQuickCalc.style.display === '') {
                bmiQuickCalc.style.display = 'block';
                toggleBmiCalcBtn.textContent = '✖ Đóng công cụ tính';
            } else {
                bmiQuickCalc.style.display = 'none';
                toggleBmiCalcBtn.textContent = '🧮 Chưa biết BMI? Tính nhanh tại đây';
            }
        });
    }

    if (btnCalculateBmi && calcHeight && calcWeight && bmiInput) {
        btnCalculateBmi.addEventListener('click', function () {
            const heightCm = parseFloat(calcHeight.value);
            const weightKg = parseFloat(calcWeight.value);

            if (isNaN(heightCm) || heightCm <= 0 || isNaN(weightKg) || weightKg <= 0) {
                if (bmiCalcResult) {
                    bmiCalcResult.innerHTML = '<span style="color: #dc2626;">⚠️ Vui lòng nhập chiều cao (cm) và cân nặng (kg) hợp lệ.</span>';
                }
                return;
            }

            // Công thức: BMI = Cân nặng (kg) / [Chiều cao (m)]^2
            const heightM = heightCm / 100.0;
            const calculatedBmi = weightKg / (heightM * heightM);
            const roundedBmi = Math.round(calculatedBmi * 10) / 10;

            // Tự động điền vào ô BMI chính
            bmiInput.value = roundedBmi;
            bmiInput.classList.remove('is-invalid');
            const errDiv = document.getElementById('bmiError');
            if (errDiv) errDiv.textContent = '';

            // Đánh giá thể trạng
            let statusText = '';
            if (roundedBmi < 18.5) statusText = 'Gầy / Thiếu cân';
            else if (roundedBmi < 23.0) statusText = 'Bình thường (Chuẩn Châu Á)';
            else if (roundedBmi < 25.0) statusText = 'Thừa cân / Tiền béo phì';
            else statusText = 'Béo phì';

            if (bmiCalcResult) {
                bmiCalcResult.innerHTML = `✅ BMI tính được: <strong>${roundedBmi} kg/m²</strong> (${statusText}). Đã tự động điền vào ô BMI!`;
            }
        });
    }

    // 3. HIỆU ỨNG THANH TIẾN TRÌNH XÁC SUẤT TRÊN TRANG KẾT QUẢ
    const probFill = document.querySelector('.prob-bar-fill');
    if (probFill) {
        const targetWidth = probFill.style.width;
        probFill.style.width = '0%';
        setTimeout(function () {
            probFill.style.width = targetWidth;
        }, 150);
    }
});

