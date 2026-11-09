"""
Build script: Generates standalone Cloudflare Pages deployment bundle
"""
import os
import shutil
import json
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(BASE_DIR, 'cloudflare_deploy')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

os.makedirs(DEPLOY_DIR, exist_ok=True)

# 1. Đọc metrics.json
with open(METRICS_PATH, 'r', encoding='utf-8') as f:
    metrics_data = json.load(f)

# 2. Khởi tạo Jinja2 Environment
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

def url_for(endpoint, **values):
    if endpoint == 'static':
        return '/static/' + values.get('filename', '')
    if endpoint == 'home':
        return '/'
    if endpoint == 'predict':
        return '/predict'
    if endpoint == 'about':
        return '/about'
    if endpoint == 'api_metrics':
        return '/api/metrics'
    return '/'

env.globals['url_for'] = url_for

# 3. Render index.html
t_index = env.get_template('index.html')
h_index = t_index.render(best_accuracy='79.9', request={'path': '/'})
with open(os.path.join(DEPLOY_DIR, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(h_index)
print("[BUILD] Đã tạo cloudflare_deploy/index.html")

# 4. Render predict.html
t_predict = env.get_template('predict.html')
h_predict = t_predict.render(error=None, form_data={}, request={'path': '/predict'})
with open(os.path.join(DEPLOY_DIR, 'predict.html'), 'w', encoding='utf-8') as f:
    f.write(h_predict)
print("[BUILD] Đã tạo cloudflare_deploy/predict.html")

# 5. Render about.html
t_about = env.get_template('about.html')
h_about = t_about.render(metrics=metrics_data, request={'path': '/about'})
with open(os.path.join(DEPLOY_DIR, 'about.html'), 'w', encoding='utf-8') as f:
    f.write(h_about)
print("[BUILD] Đã tạo cloudflare_deploy/about.html")

# 6. Render result_template.html (dùng placeholder để _worker.js điền nội dung)
dummy_result = {
    'prediction': 0,
    'label_text': '__LABEL_TEXT__',
    'prob_negative': '__PROB_NEG__',
    'prob_positive': '__PROB_POS__',
    'confidence': '__CONFIDENCE__',
    'glucose': '__GLUCOSE__',
    'bmi': '__BMI__',
    'age': '__AGE__',
    'interpretations': [
        {'feature': '__FEAT_0__', 'value': '__VAL_0__', 'status': 'normal', 'note': '__NOTE_0__'}
    ],
    'recommendations': ['__REC_0__']
}
t_result = env.get_template('result.html')
h_result_raw = t_result.render(result=dummy_result, request={'path': '/result'})
with open(os.path.join(DEPLOY_DIR, 'result_template.html'), 'w', encoding='utf-8') as f:
    f.write(h_result_raw)
print("[BUILD] Đã tạo cloudflare_deploy/result_template.html")

# 7. Sao chép static files
deploy_static = os.path.join(DEPLOY_DIR, 'static')
if os.path.exists(deploy_static):
    shutil.rmtree(deploy_static)
shutil.copytree(STATIC_DIR, deploy_static)
print("[BUILD] Đã sao chép static/ (CSS, JS, Images)")

# 8. Sao chép metrics.json
with open(os.path.join(DEPLOY_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, ensure_ascii=False, indent=2)
print("[BUILD] Đã lưu cloudflare_deploy/metrics.json")

