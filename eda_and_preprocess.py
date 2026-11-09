"""
Script: eda_and_preprocess.py
Mô tả: Thực hiện Phân tích Dữ liệu Khám phá (EDA), Trực quan hóa và Tiền xử lý dữ liệu.
Đề tài: Dự đoán nguy cơ tiểu đường dựa trên Glucose, BMI, Tuổi.
"""

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Cấu hình giao diện biểu đồ
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# Thư mục lưu biểu đồ
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'static', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. ĐỌC DỮ LIỆU
data_path = os.path.join(os.path.dirname(__file__), 'data', 'diabetes.csv')
df = pd.read_csv(data_path)

print("=" * 60)
print("1. KHẢO SÁT DỮ LIỆU BAN ĐẦU")
print("=" * 60)
print(f"Tổng số dòng: {df.shape[0]}, Tổng số cột: {df.shape[1]}")
features = ['Glucose', 'BMI', 'Age']
target = 'Outcome'
df_sub = df[features + [target]].copy()

print("\nNăm dòng đầu tiên của 3 đặc trưng đã chọn:")
print(df_sub.head())

# 2. PHÁT HIỆN & XỬ LÝ GIÁ TRỊ 0 BẤT THƯỜNG
print("\n" + "=" * 60)
print("2. PHÁT HIỆN GIÁ TRỊ 0 BẤT THƯỜNG")
print("=" * 60)
for col in ['Glucose', 'BMI']:
    zeros = (df_sub[col] == 0).sum()
    pct = zeros / len(df_sub) * 100
    print(f"- Cột '{col}': {zeros} giá trị bằng 0 ({pct:.2f}%)")

# Gán giá trị 0 thành NaN để phân tích chính xác
df_cleaned = df_sub.copy()
df_cleaned['Glucose'] = df_cleaned['Glucose'].replace(0, np.nan)
df_cleaned['BMI'] = df_cleaned['BMI'].replace(0, np.nan)

# 3. TRỰC QUAN HÓA DỮ LIỆU (EDA)
print("\n" + "=" * 60)
print("3. TẠO BIỂU ĐỒ TRỰC QUAN HÓA (LƯU VÀO static/images/)")
print("=" * 60)

# 3.1. Biểu đồ phân bố nhãn Outcome
plt.figure(figsize=(7, 4.5), dpi=300)
colors = ['#3182ce', '#e53e3e']
ax = sns.countplot(data=df_cleaned, x='Outcome', palette=colors, hue='Outcome', legend=False)
plt.title('Phân bố nhãn mục tiêu (0: Không tiểu đường, 1: Tiểu đường)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Nhãn Outcome', fontsize=11)
plt.ylabel('Số lượng mẫu', fontsize=11)
plt.xticks([0, 1], ['Không tiểu đường (0)', 'Tiểu đường (1)'])

total = len(df_cleaned)
for p in ax.patches:
    count = int(p.get_height())
    pct = count / total * 100
    ax.annotate(f'{count} ({pct:.1f}%)', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')

plt.tight_layout()
outcome_img_path = os.path.join(IMAGES_DIR, 'outcome_distribution.png')
plt.savefig(outcome_img_path)
plt.close()
print(f"Đã lưu: {outcome_img_path}")

# 3.2. Biểu đồ phân bố (Histogram + KDE) 3 đặc trưng theo Outcome
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
feature_names = {
    'Glucose': 'Glucose (Đường huyết - mg/dL)',
    'BMI': 'BMI (Chỉ số khối cơ thể - kg/m²)',
    'Age': 'Age (Tuổi - năm)'
}

for i, col in enumerate(features):
    sns.histplot(data=df_cleaned, x=col, hue='Outcome', kde=True,
                 palette={0: '#3182ce', 1: '#e53e3e'}, ax=axes[i],
                 bins=25, alpha=0.45, edgecolor='none')
    axes[i].set_title(feature_names[col], fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Giá trị', fontsize=10)
    axes[i].set_ylabel('Tần suất' if i == 0 else '', fontsize=10)
    axes[i].legend(title='Nhãn', labels=['Tiểu đường (1)', 'Không tiểu đường (0)'])

plt.suptitle('Phân bố 3 chỉ số theo tình trạng tiểu đường', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
dist_img_path = os.path.join(IMAGES_DIR, 'feature_distributions.png')
plt.savefig(dist_img_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {dist_img_path}")

# 3.3. Biểu đồ Ma trận tương quan (Correlation Heatmap)
plt.figure(figsize=(6, 5), dpi=300)
corr = df_cleaned.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', vmin=-0.1, vmax=0.6,
            linewidths=1, square=True, cbar_kws={'shrink': 0.8})
plt.title('Ma trận tương quan Pearson giữa các biến', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
corr_img_path = os.path.join(IMAGES_DIR, 'correlation_matrix.png')
plt.savefig(corr_img_path)
plt.close()
print(f"Đã lưu: {corr_img_path}")

# 3.4. Biểu đồ Boxplot so sánh ngoại lai và trung vị
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
for i, col in enumerate(features):
    sns.boxplot(data=df_cleaned, x='Outcome', y=col, palette=['#63b3ed', '#feb2b2'], ax=axes[i], hue='Outcome', legend=False)
    axes[i].set_title(feature_names[col], fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Outcome', fontsize=10)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Không (0)', 'Có (1)'])
    axes[i].set_ylabel('Giá trị' if i == 0 else '', fontsize=10)

plt.suptitle('So sánh giá trị trung vị & ngoại lai giữa hai nhóm', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
box_img_path = os.path.join(IMAGES_DIR, 'feature_boxplots.png')
plt.savefig(box_img_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {box_img_path}")

# 4. QUY TRÌNH CHIA DỮ LIỆU & TIỀN XỬ LÝ CHỐNG DATA LEAKAGE
print("\n" + "=" * 60)
print("4. CHIA TẬP TRAIN/TEST (80/20) VỚI STRATIFY & KIỂM TRA DATA LEAKAGE")
print("=" * 60)

# Tách đặc trưng X và nhãn y
X = df_sub[features].copy()
y = df_sub[target].copy()

# Thay thế giá trị 0 thành np.nan trong X
X['Glucose'] = X['Glucose'].replace(0, np.nan)
X['BMI'] = X['BMI'].replace(0, np.nan)

# Phân chia dữ liệu 80% train, 20% test (stratify để bảo toàn tỷ lệ nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Kích thước X_test:  {X_test.shape}, y_test:  {y_test.shape}")

print("\nTỷ lệ nhãn trong tập Train:")
print(y_train.value_counts(normalize=True).round(4) * 100)
print("\nTỷ lệ nhãn trong tập Test:")
print(y_test.value_counts(normalize=True).round(4) * 100)

# 4.1. Fit Imputer CHỈ TRÊN TẬP TRAIN (Chống data leakage)
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)

print("\n--- GIÁ TRỊ TRUNG VỊ ĐƯỢC HỌC TỪ TẬP TRAIN (FIT) ---")
for col, val in zip(features, imputer.statistics_):
    print(f"- {col}: {val:.2f}")

# Biến đổi (Transform) cho cả Train và Test
X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=features, index=X_train.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=features, index=X_test.index)

# 4.2. Fit Scaler CHỈ TRÊN TẬP TRAIN (Chống data leakage)
scaler = StandardScaler()
scaler.fit(X_train_imp)

print("\n--- THÔNG SỐ CHUẨN HÓA ĐƯỢC HỌC TỪ TẬP TRAIN (FIT) ---")
for col, m, s in zip(features, scaler.mean_, scaler.scale_):
    print(f"- {col}: Mean = {m:.2f}, Std = {s:.2f}")

X_train_scaled = scaler.transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("\nKiểm tra dữ liệu sau chuẩn hóa (Mean ~ 0, Std ~ 1 trên tập Train):")
print(f"- Train Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"- Train Std:  {X_train_scaled.std(axis=0).round(4)}")
print(f"- Test Mean (phải khác 0 do dùng phân bố của Train): {X_test_scaled.mean(axis=0).round(4)}")

print("\nHoàn tất toàn bộ bước Tiền xử lý dữ liệu chuẩn Machine Learning!")
