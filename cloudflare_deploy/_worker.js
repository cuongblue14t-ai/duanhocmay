/**
 * Cloudflare Pages Advanced Mode Worker (_worker.js)
 * DiaPredict ML - Hệ Thống Hỗ Trợ Dự Đoán Nguy Cơ Tiểu Đường
 * Triển khai trực tiếp toàn bộ mô hình Decision Tree Pipeline & RESTful API trên Cloudflare Edge.
 */

// 1. Tham số tiền xử lý chuẩn hóa từ tập Train
const IMPUTER_MEDIANS = {
    Glucose: 117.0,
    BMI: 32.4,
    Age: 29.0
};

const SCALER_MEANS = {
    Glucose: 121.67100977198697,
    BMI: 32.44820846905537,
    Age: 33.36644951140065
};

const SCALER_SCALES = {
    Glucose: 29.979350764655706,
    BMI: 6.81856216148434,
    Age: 11.82379790254742
};

// 2. Hàm dự đoán từ Cây Quyết Định (Decision Tree depth=4, 13 leaves)
function predictTree(g_val, b_val, a_val) {
    // Tiền xử lý: Điền khuyết (Imputation)
    const g_imp = (g_val === 0 || isNaN(g_val)) ? IMPUTER_MEDIANS.Glucose : g_val;
    const b_imp = (b_val === 0 || isNaN(b_val)) ? IMPUTER_MEDIANS.BMI : b_val;
    const a_imp = a_val;

    // Chuẩn hóa thang đo (StandardScaler)
    const g_s = (g_imp - SCALER_MEANS.Glucose) / SCALER_SCALES.Glucose;
    const b_s = (b_imp - SCALER_MEANS.BMI) / SCALER_SCALES.BMI;
    const a_s = (a_imp - SCALER_MEANS.Age) / SCALER_SCALES.Age;

    // Phân lớp theo Cây Quyết Định
    if (g_s <= 1.0950533747673035) {
        if (b_s <= -0.7476955354213715) {
            if (g_s <= 0.3945712596178055) {
                if (b_s <= -0.8136918544769287) {
                    return { label: 0, prob_0: 0.9904761904761905, prob_1: 0.009523809523809525 };
                } else {
                    return { label: 0, prob_0: 0.8571428571428571, prob_1: 0.14285714285714285 };
                }
            } else {
                if (b_s <= -0.9823491275310516) {
                    return { label: 0, prob_0: 0.75, prob_1: 0.25 };
                } else {
                    return { label: 0, prob_0: 0.875, prob_1: 0.125 };
                }
            }
        } else {
            if (a_s <= -0.4115809053182602) {
                if (g_s <= 0.19443350285291672) {
                    return { label: 0, prob_0: 0.8827586206896552, prob_1: 0.11724137931034483 };
                } else {
                    return { label: 0, prob_0: 0.55, prob_1: 0.45 };
                }
            } else {
                if (g_s <= -0.4726923555135727) {
                    return { label: 0, prob_0: 0.7464788732394366, prob_1: 0.2535211267605634 };
                } else {
                    return { label: 1, prob_0: 0.44274809160305345, prob_1: 0.5572519083969466 };
                }
            }
        }
    } else {
        if (a_s <= 2.463975667953491) {
            if (g_s <= 1.2284786105155945) {
                if (a_s <= 0.3918834272772074) {
                    return { label: 1, prob_0: 0.4, prob_1: 0.6 };
                } else {
                    return { label: 1, prob_0: 0.0, prob_1: 1.0 };
                }
            } else {
                if (g_s <= 1.3952600359916687) {
                    return { label: 1, prob_0: 0.0, prob_1: 1.0 };
                } else {
                    return { label: 1, prob_0: 0.13432835820895522, prob_1: 0.8656716417910447 };
                }
            }
        } else {
            return { label: 0, prob_0: 0.6, prob_1: 0.4 };
        }
    }
}

// 3. Hàm kiểm tra hợp lệ dữ liệu nhập
function validateInput(glucoseRaw, bmiRaw, ageRaw) {
    const glucose = parseFloat(glucoseRaw);
    const bmi = parseFloat(bmiRaw);
    const age = parseFloat(ageRaw);

    if (isNaN(glucose) || isNaN(bmi) || isNaN(age)) {
        return { values: null, error: 'Vui lòng nhập đầy đủ các chỉ số dưới dạng số hợp lệ.' };
    }
    if (glucose < 0 || glucose > 500) {
        return { values: null, error: 'Chỉ số Glucose phải nằm trong khoảng từ 0 đến 500 mg/dL.' };
    }
    if (bmi < 0 || bmi > 100) {
        return { values: null, error: 'Chỉ số BMI phải nằm trong khoảng từ 0 đến 100 kg/m².' };
    }
    if (age < 1 || age > 125) {
        return { values: null, error: 'Độ tuổi phải nằm trong khoảng từ 1 đến 125 tuổi.' };
    }

    return {
        values: { Glucose: glucose, BMI: bmi, Age: age },
        error: null
    };
}

// 4. Hàm sinh phân tích lâm sàng và khuyến nghị
function generateInterpretation(glucose, bmi, age, prediction) {
    const interpretations = [];

    // Glucose
    if (glucose === 0) {
        interpretations.push({
            feature: 'Glucose (Đường huyết)',
            value: 'Chưa đo (0 mg/dL)',
            status: 'warning',
            note: 'Hệ thống đã tự động ước lượng theo giá trị trung vị của tập dữ liệu y khoa.'
        });
    } else if (glucose < 100) {
        interpretations.push({
            feature: 'Glucose (Đường huyết)',
            value: `${glucose.toFixed(1)} mg/dL`,
            status: 'normal',
            note: 'Trong ngưỡng an toàn bình thường (dưới 100 mg/dL).'
        });
    } else if (glucose <= 125) {
        interpretations.push({
            feature: 'Glucose (Đường huyết)',
            value: `${glucose.toFixed(1)} mg/dL`,
            status: 'warning',
            note: 'Ở mức Tiền đái tháo đường (100 - 125 mg/dL). Cần chú ý điều chỉnh chế độ ăn uống.'
        });
    } else {
        interpretations.push({
            feature: 'Glucose (Đường huyết)',
            value: `${glucose.toFixed(1)} mg/dL`,
            status: 'danger',
            note: 'Đường huyết ở mức cao (≥ 126 mg/dL). Đây là dấu hiệu cảnh báo nguy cơ tiểu đường rõ rệt.'
        });
    }

    // BMI
    if (bmi === 0) {
        interpretations.push({
            feature: 'BMI (Thể trạng)',
            value: 'Chưa đo (0 kg/m²)',
            status: 'warning',
            note: 'Hệ thống đã tự động điền khuyết theo trung vị thể trạng dân số mẫu.'
        });
    } else if (bmi < 18.5) {
        interpretations.push({
            feature: 'BMI (Thể trạng)',
            value: `${bmi.toFixed(1)} kg/m²`,
            status: 'normal',
            note: 'Thể trạng nhẹ cân/gầy so với chiều cao.'
        });
    } else if (bmi < 23.0) {
        interpretations.push({
            feature: 'BMI (Thể trạng)',
            value: `${bmi.toFixed(1)} kg/m²`,
            status: 'normal',
            note: 'Cân nặng chuẩn lý tưởng (18.5 - 22.9 kg/m² theo chuẩn WHO Châu Á).'
        });
    } else if (bmi < 25.0) {
        interpretations.push({
            feature: 'BMI (Thể trạng)',
            value: `${bmi.toFixed(1)} kg/m²`,
            status: 'warning',
            note: 'Tiền béo phì / Thừa cân (23.0 - 24.9 kg/m²). Tăng nguy cơ kháng insulin.'
        });
    } else {
        interpretations.push({
            feature: 'BMI (Thể trạng)',
            value: `${bmi.toFixed(1)} kg/m²`,
            status: 'danger',
            note: `Béo phì (${bmi.toFixed(1)} kg/m²). Lượng mỡ thừa là yếu tố thúc đẩy đái tháo đường typ 2.`
        });
    }

    // Tuổi
    if (age < 35) {
        interpretations.push({
            feature: 'Độ tuổi',
            value: `${Math.round(age)} tuổi`,
            status: 'normal',
            note: 'Nhóm tuổi trẻ, chuyển hóa cơ bản thường hoạt động tốt hơn.'
        });
    } else if (age < 50) {
        interpretations.push({
            feature: 'Độ tuổi',
            value: `${Math.round(age)} tuổi`,
            status: 'warning',
            note: 'Lứa tuổi trung niên (35 - 49), bắt đầu tích lũy nguy cơ rối loạn chuyển hóa.'
        });
    } else {
        interpretations.push({
            feature: 'Độ tuổi',
            value: `${Math.round(age)} tuổi`,
            status: 'warning',
            note: 'Từ 50 tuổi trở lên là nhóm có nguy cơ tự nhiên cao hơn, cần sàng lọc đường huyết định kỳ.'
        });
    }

    // Khuyến nghị
    const recommendations = [];
    if (prediction === 1) {
        recommendations.push('Nên sớm đến các cơ sở y tế hoặc bệnh viện để thực hiện xét nghiệm HbA1c và kiểm tra đường huyết tĩnh mạch chính xác.');
        recommendations.push('Hạn chế tối đa nước ngọt có gas, bánh kẹo ngọt và các loại tinh bột tinh chế (cơm trắng, bánh mì trắng).');
        recommendations.push('Tăng cường chất xơ từ rau xanh, các loại hạt và duy trì hoạt động thể chất tối thiểu 30 phút mỗi ngày.');
        recommendations.push('Nếu chỉ số BMI cao, việc giảm 5-10% trọng lượng cơ thể sẽ cải thiện rõ rệt độ nhạy của insulin.');
    } else {
        recommendations.push('Duy trì lối sống lành mạnh, cân bằng dinh dưỡng và thói quen vận động đều đặn.');
        recommendations.push('Khám sức khỏe tổng quát định kỳ mỗi 6 - 12 tháng để theo dõi các chỉ số sinh hóa máu.');
        recommendations.push('Uống đủ nước, ngủ đủ giấc và kiểm soát căng thẳng (stress) để duy trì sự ổn định của hệ nội tiết.');
    }

    return { interpretations, recommendations };
}

// 5. Hàm tính toán kết quả đầy đủ
function calculateFullResult(glucose, bmi, age) {
    const treeOut = predictTree(glucose, bmi, age);
    const predLabel = treeOut.label;
    const probNeg = parseFloat((treeOut.prob_0 * 100).toFixed(1));
    const probPos = parseFloat((treeOut.prob_1 * 100).toFixed(1));
    const confidence = predLabel === 1 ? probPos : probNeg;
    const labelText = predLabel === 1
        ? 'Cảnh báo: Có nguy cơ mắc bệnh tiểu đường'
        : 'Nguy cơ thấp: Không có dấu hiệu tiểu đường';

    const { interpretations, recommendations } = generateInterpretation(glucose, bmi, age, predLabel);

    return {
        prediction: predLabel,
        label_text: labelText,
        prob_negative: probNeg.toFixed(1),
        prob_positive: probPos.toFixed(1),
        confidence: confidence.toFixed(1),
        glucose: glucose,
        bmi: bmi,
        age: Math.round(age),
        interpretations,
        recommendations
    };
}

// 6. Hàm render trang HTML Kết quả
function renderResultHtml(templateHtml, result) {
    const isDanger = result.prediction === 1;
    const heroCardClass = isDanger ? 'card-danger' : 'card-success';
    const fillClass = isDanger ? 'fill-danger' : 'fill-success';
    const resultTag = isDanger ? 'NHÃN DỰ ĐOÁN: 1 (DƯƠNG TÍNH)' : 'NHÃN DỰ ĐOÁN: 0 (ÂM TÍNH)';

    const iconSvg = isDanger
        ? `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
        : `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

    // Render cards phân tích
    const metricsHtml = result.interpretations.map(item => `
        <div class="metric-card metric-${item.status}">
            <div class="metric-card-top">
                <span class="metric-card-name">${item.feature}</span>
                <span class="status-pill pill-${item.status}">
                    ${item.status === 'normal' ? 'Bình thường' : item.status === 'warning' ? 'Cần lưu ý' : 'Nguy cơ cao'}
                </span>
            </div>
            <div class="metric-card-value">${item.value}</div>
            <p class="metric-card-note">${item.note}</p>
        </div>
    `).join('');

    // Render recommendations
    const recsHtml = result.recommendations.map(rec => `
        <li class="recommendation-item">
            <span class="rec-icon">📌</span>
            <span>${rec}</span>
        </li>
    `).join('');

    let output = templateHtml;

    // Thay thế hero card
    output = output.replace('card-success', heroCardClass);
    output = output.replace(/<svg width="48" height="48"[\s\S]*?<\/svg>/, iconSvg);
    output = output.replace(/NHÃN DỰ ĐOÁN: [01] \((?:ÂM|DƯƠNG) TÍNH\)/, resultTag);
    output = output.replace('__LABEL_TEXT__', result.label_text);
    output = output.replaceAll('__CONFIDENCE__', result.confidence);
    output = output.replace('fill-success', fillClass);
    output = output.replaceAll('__PROB_NEG__', result.prob_negative);
    output = output.replaceAll('__PROB_POS__', result.prob_positive);

    // Thay thế metrics grid
    output = output.replace(/<div class="metrics-grid">[\s\S]*?<\/div>\s*<\/div>/, `<div class="metrics-grid">${metricsHtml}</div></div>`);

    // Thay thế recommendations
    output = output.replace(/<ul class="recommendation-list">[\s\S]*?<\/ul>/, `<ul class="recommendation-list">${recsHtml}</ul>`);

    // Thay thế Chart.js values
    output = output.replace('data: [__PROB_NEG__, __PROB_POS__]', `data: [${result.prob_negative}, ${result.prob_positive}]`);
    output = output.replace('data: [__GLUCOSE__, __BMI__, __AGE__]', `data: [${result.glucose}, ${result.bmi}, ${result.age}]`);

    return output;
}

// 7. Worker Fetch Handler
export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        const path = url.pathname;
        const method = request.method;

        // Trang chủ
        if (path === '/' || path === '/index.html') {
            return env.ASSETS.fetch(new Request(new URL('/index.html', request.url).toString(), { method: 'GET' }));
        }

        // Trang About
        if (path === '/about' || path === '/about.html') {
            return env.ASSETS.fetch(new Request(new URL('/about.html', request.url).toString(), { method: 'GET' }));
        }

        // Trang Predict (GET)
        if ((path === '/predict' || path === '/predict.html') && method === 'GET') {
            return env.ASSETS.fetch(new Request(new URL('/predict.html', request.url).toString(), { method: 'GET' }));
        }

        // API Metrics (JSON)
        if (path === '/api/metrics') {
            const metricsRes = await env.ASSETS.fetch(new Request(new URL('/metrics.json', request.url).toString(), { method: 'GET' }));
            const data = await metricsRes.json();
            return new Response(JSON.stringify({
                status: 'success',
                data: data
            }), {
                headers: {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Access-Control-Allow-Origin': '*'
                }
            });
        }

        // API Predict (POST JSON)
        if (path === '/api/predict') {
            if (method !== 'POST') {
                return new Response(JSON.stringify({ status: 'error', message: 'Phương thức yêu cầu phải là POST' }), {
                    status: 405,
                    headers: { 'Content-Type': 'application/json; charset=utf-8' }
                });
            }

            let body;
            try {
                body = await request.json();
            } catch (e) {
                return new Response(JSON.stringify({ status: 'error', message: 'Yêu cầu định dạng Content-Type: application/json' }), {
                    status: 400,
                    headers: { 'Content-Type': 'application/json; charset=utf-8' }
                });
            }

            const { values, error } = validateInput(body.glucose, body.bmi, body.age);
            if (error) {
                return new Response(JSON.stringify({ status: 'error', message: error }), {
                    status: 400,
                    headers: { 'Content-Type': 'application/json; charset=utf-8' }
                });
            }

            const result = calculateFullResult(values.Glucose, values.BMI, values.Age);

            return new Response(JSON.stringify({
                status: 'success',
                input: {
                    glucose: values.Glucose,
                    bmi: values.BMI,
                    age: values.Age
                },
                prediction: result.prediction,
                label: result.prediction === 1 ? 'Nguy cơ tiểu đường (Dương tính)' : 'Không tiểu đường (Âm tính)',
                probabilities: {
                    negative_class_0_pct: parseFloat(result.prob_negative),
                    positive_class_1_pct: parseFloat(result.prob_positive)
                },
                confidence_score_pct: parseFloat(result.confidence),
                interpretations: result.interpretations,
                recommendations: result.recommendations,
                disclaimer: 'Kết quả từ hệ thống chỉ mang tính chất tham khảo học tập, không thay thế chẩn đoán y khoa từ bác sĩ.'
            }), {
                headers: {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Access-Control-Allow-Origin': '*'
                }
            });
        }

        // Xử lý Form POST /predict -> Trả về giao diện Kết Quả
        if (path === '/predict' && method === 'POST') {
            let glucoseRaw = '';
            let bmiRaw = '';
            let ageRaw = '';

            try {
                const formData = await request.formData();
                glucoseRaw = formData.get('glucose');
                bmiRaw = formData.get('bmi');
                ageRaw = formData.get('age');
            } catch (e) {
                // Thử đọc urlencoded text
                const text = await request.text();
                const params = new URLSearchParams(text);
                glucoseRaw = params.get('glucose');
                bmiRaw = params.get('bmi');
                ageRaw = params.get('age');
            }

            const { values, error } = validateInput(glucoseRaw, bmiRaw, ageRaw);
            if (error) {
                const predictRes = await env.ASSETS.fetch(new Request(new URL('/predict.html', request.url).toString(), { method: 'GET' }));
                let predictHtml = await predictRes.text();
                const alertHtml = `
                    <div class="alert alert-danger" role="alert">
                        <div class="alert-icon">❌</div>
                        <div><strong>Lỗi nhập liệu:</strong> ${error}</div>
                    </div>
                `;
                predictHtml = predictHtml.replace('<div class="page-title-block', alertHtml + '<div class="page-title-block');
                return new Response(predictHtml, {
                    headers: { 'Content-Type': 'text/html; charset=utf-8' }
                });
            }

            const result = calculateFullResult(values.Glucose, values.BMI, values.Age);
            const tmplRes = await env.ASSETS.fetch(new Request(new URL('/result_template.html', request.url).toString(), { method: 'GET' }));
            const tmplHtml = await tmplRes.text();
            const renderedHtml = renderResultHtml(tmplHtml, result);

            return new Response(renderedHtml, {
                headers: { 'Content-Type': 'text/html; charset=utf-8' }
            });
        }

        // Xem kết quả qua GET /result?glucose=...&bmi=...&age=...
        if (path === '/result' || path === '/result.html') {
            const glucose = url.searchParams.get('glucose');
            const bmi = url.searchParams.get('bmi');
            const age = url.searchParams.get('age');

            if (glucose && bmi && age) {
                const { values, error } = validateInput(glucose, bmi, age);
                if (!error) {
                    const result = calculateFullResult(values.Glucose, values.BMI, values.Age);
                    const tmplRes = await env.ASSETS.fetch(new Request(new URL('/result_template.html', request.url).toString(), { method: 'GET' }));
                    const tmplHtml = await tmplRes.text();
                    const renderedHtml = renderResultHtml(tmplHtml, result);
                    return new Response(renderedHtml, {
                        headers: { 'Content-Type': 'text/html; charset=utf-8' }
                    });
                }
            }
            return Response.redirect(new URL('/predict', request.url).toString(), 302);
        }

        // Mặc định nạp static assets từ Cloudflare Pages
        return env.ASSETS.fetch(request);
    }
};
