"""One-shot builder: writes notebooks/03_gradcam_analysis.ipynb from scratch."""
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "03_gradcam_analysis.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# === Section 0: Title ===
cells.append(nbf.v4.new_markdown_cell("""# 03 — Phân tích Grad-CAM cho CIFAKE

**Mục tiêu:** Hiểu *vì sao* các model phân biệt ảnh thật vs ảnh AI bằng cách nhìn vào vùng pixel mà model **chú ý** (attention) khi đưa ra quyết định.

Notebook này dựa trên dữ liệu đã được sinh sẵn ở `outputs/gradcam_batch/` (xem `scripts/generate_gradcam_batch.py`)."""))

# === Section 1: Theory ===
cells.append(nbf.v4.new_markdown_cell("""## 1. Grad-CAM là gì?

**Grad-CAM (Gradient-weighted Class Activation Mapping)** trả lời câu hỏi: *"Pixel nào trong ảnh đẩy điểm số lớp dự đoán lên cao?"*

Cách hoạt động (tóm tắt):
1. Cho ảnh qua CNN → lấy **feature map** ở 1 conv layer cuối (gọi là `A`).
2. Tính **gradient** của điểm số lớp dự đoán theo `A`.
3. Trung bình gradient theo trục không gian → **trọng số** cho từng kênh.
4. Tổ hợp tuyến tính các kênh `A` với trọng số → bản đồ nhiệt 2D → ReLU → upsample về kích thước ảnh gốc.

→ Vùng **"nóng" (đỏ)** = vùng model dựa vào để ra quyết định. Vùng **"lạnh" (xanh)** = model bỏ qua.

> Trong project này: target layer = conv block cuối cùng của mỗi kiến trúc (xem `scripts/generate_gradcam_batch.py` → `MODEL_REGISTRY`)."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = PROJECT_ROOT / "outputs" / "gradcam_batch"
df = pd.read_csv(DATA_DIR / "metadata.csv")
print("Shape:", df.shape)
print()
print("Accuracy theo model (trên 100 mẫu test đầu tiên):")
print(df.groupby("model")["correct"].mean().round(3))
df.head()"""))

# === Section 2: Correct samples ===
cells.append(nbf.v4.new_markdown_cell("""## 2. Samples đúng — model "nhìn" vào đâu?

Lấy 4 ảnh REAL và 4 ảnh FAKE mà **DenseNet121 phân loại đúng** rồi hiển thị heatmap overlay (heatmap chồng lên ảnh gốc)."""))

cells.append(nbf.v4.new_code_cell("""def show_overlays(rows, title):
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.8))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        original = np.array(Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB"))
        heatmap = np.load(PROJECT_ROOT / row["heatmap_path"])
        ax.imshow(original)
        ax.imshow(heatmap, cmap="jet", alpha=0.45)
        ax.set_title(
            f"T:{row['true_label']} P:{row['pred_label']}\\np(REAL)={row['prob_real']:.2f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

dense = df[df["model"] == "densenet121"]
correct_real = dense[(dense["true_label"] == "REAL") & (dense["correct"] == 1)].head(4)
correct_fake = dense[(dense["true_label"] == "FAKE") & (dense["correct"] == 1)].head(4)

show_overlays(correct_real, "DenseNet121 — REAL đúng")
show_overlays(correct_fake, "DenseNet121 — FAKE đúng")"""))

cells.append(nbf.v4.new_markdown_cell("""**Nhận xét cần điền sau khi chạy:**
- Vùng "nóng" trên ảnh REAL thường rơi vào đâu? (chủ thể / nền / cạnh?)
- Vùng "nóng" trên ảnh FAKE có khác biệt rõ với ảnh REAL không?
- Có ảnh nào model dự đúng nhưng nhìn vào vùng "sai" (vd: nhìn vào nền) — dấu hiệu *shortcut learning*?"""))

# === Section 3: Wrong cases ===
cells.append(nbf.v4.new_markdown_cell("""## 3. Các trường hợp dự đoán SAI

Đây là phần quan trọng nhất: model thất bại khi nào, và *vì sao*?"""))

cells.append(nbf.v4.new_code_cell("""wrong = dense[dense["correct"] == 0].copy()
wrong["confidence"] = (wrong["prob_real"] - 0.5).abs()
print(f"Số case sai của DenseNet121 (trong 100 mẫu): {len(wrong)}")

if len(wrong) > 0:
    print()
    print("Top case sai TỰ TIN NHẤT (model rất chắc nhưng sai):")
    top_wrong = wrong.sort_values("confidence", ascending=False).head(min(4, len(wrong)))
    show_overlays(top_wrong, "DenseNet121 — sai mà rất tự tin")
    print()
    print("Top case sai MƠ HỒ NHẤT (xác suất sát 0.5):")
    border = wrong.sort_values("confidence").head(min(4, len(wrong)))
    show_overlays(border, "DenseNet121 — sai và mơ hồ")
else:
    print("Không có case sai trong 100 mẫu đầu — thử mở rộng dataset.")"""))

cells.append(nbf.v4.new_markdown_cell("""**Câu hỏi để trả lời sau khi chạy:**
1. Loại sai nào phổ biến hơn — false-positive (FAKE bị nhận thành REAL) hay false-negative?
2. Với case sai "tự tin": Grad-CAM có chỉ vào vùng *bất thường* (vd: artifact của ảnh FAKE) hay vào vùng vô nghĩa?
3. Với case sai "mơ hồ": vùng attention có bị phân tán không?"""))

# === Section 4: Cross-model comparison ===
cells.append(nbf.v4.new_markdown_cell("""## 4. So sánh 4 model trên cùng ảnh

4 kiến trúc khác nhau (SimpleCNN / ResNet18 / EfficientNet-B0 / DenseNet121) có chú ý cùng một vùng không?"""))

cells.append(nbf.v4.new_code_cell("""MODELS = ["simple_cnn", "resnet18", "efficientnet_b0", "densenet121"]

def compare_models_on_image(image_path):
    fig, axes = plt.subplots(1, len(MODELS) + 1, figsize=(2.4 * (len(MODELS) + 1), 2.6))
    original = np.array(Image.open(PROJECT_ROOT / image_path).convert("RGB"))
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")
    true_label = None
    for ax, model_name in zip(axes[1:], MODELS):
        row = df[(df["model"] == model_name) & (df["image_path"] == str(image_path))].iloc[0]
        true_label = row["true_label"]
        heatmap = np.load(PROJECT_ROOT / row["heatmap_path"])
        ax.imshow(original)
        ax.imshow(heatmap, cmap="jet", alpha=0.45)
        ax.set_title(
            f"{model_name}\\nP:{row['pred_label']} ({row['prob_real']:.2f})",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(f"True label: {true_label}")
    plt.tight_layout()
    plt.show()

sample_real = df[(df["model"] == "densenet121") & (df["true_label"] == "REAL")].iloc[0]["image_path"]
sample_fake = df[(df["model"] == "densenet121") & (df["true_label"] == "FAKE")].iloc[0]["image_path"]
compare_models_on_image(Path(sample_real))
compare_models_on_image(Path(sample_fake))"""))

cells.append(nbf.v4.new_markdown_cell("""**Quan sát:** Model nông (SimpleCNN) thường có attention thô và rộng, model sâu (DenseNet) tập trung gọn hơn. Ghi nhận vào báo cáo."""))

# === Section 5: Quantitative ===
cells.append(nbf.v4.new_markdown_cell("""## 5. Phân tích định lượng heatmap

Thay vì chỉ "nhìn bằng mắt", ta đo các chỉ số:
- **coverage**: tỉ lệ pixel có giá trị ≥ 0.5 (vùng nóng chiếm bao nhiêu % ảnh)
- **entropy**: độ tản của attention (cao = trải đều, thấp = tập trung)
- **center_mass_(x,y)**: tâm khối lượng vùng nóng (0–1, 0.5 = giữa ảnh)"""))

cells.append(nbf.v4.new_code_cell("""from src.explainability.heatmap_stats import heatmap_stats

records = []
for _, row in df.iterrows():
    h = np.load(PROJECT_ROOT / row["heatmap_path"])
    stats = heatmap_stats(h)
    records.append({**row.to_dict(), **stats})

stats_df = pd.DataFrame(records)
summary = stats_df.groupby("model")[["coverage", "entropy", "peak_value"]].mean().round(3)
print("Trung bình theo model:")
print(summary)"""))

cells.append(nbf.v4.new_code_cell("""# So sánh case đúng vs sai
by_correct = stats_df.groupby(["model", "correct"])[["coverage", "entropy"]].mean().round(3)
print("Đúng (1) vs Sai (0):")
print(by_correct)"""))

cells.append(nbf.v4.new_code_cell("""# Histogram coverage cho 4 model
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for ax, model_name in zip(axes, MODELS):
    sub = stats_df[stats_df["model"] == model_name]
    ax.hist(sub[sub["correct"] == 1]["coverage"], bins=20, alpha=0.6, label="đúng")
    if (sub["correct"] == 0).any():
        ax.hist(sub[sub["correct"] == 0]["coverage"], bins=20, alpha=0.6, label="sai")
    ax.set_title(model_name)
    ax.set_xlabel("coverage")
    ax.legend()
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""**Đọc số liệu:**
- Nếu coverage trung bình của case **sai** > case **đúng** → khi model bối rối, nó "trải attention" ra nhiều vùng → đây là tín hiệu định lượng cho confidence.
- Entropy thấp + coverage thấp + accuracy cao → model học được đặc trưng định vị tốt (tập trung vào vùng có nghĩa)."""))

# === Section 6: Conclusion ===
cells.append(nbf.v4.new_markdown_cell("""## 6. Kết luận

### 6.1 Trả lời câu hỏi nghiên cứu
> *"Model học được gì để phân biệt ảnh thật vs ảnh AI?"*

**Điền sau khi đã chạy và quan sát các phần trên:**

- Trên ảnh REAL, model tập trung vào: ...
- Trên ảnh FAKE, model tập trung vào: ...
- Khác biệt định lượng giữa case đúng và case sai: ... (dẫn số từ Phần 5)

### 6.2 Hạn chế phát hiện được
- ...
- ...

### 6.3 Hướng mở rộng (cho phần "Future work" trong đồ án)
- Thử thêm phương pháp XAI khác: **LIME** (giải thích bằng perturbation), **SHAP** (đóng góp từng vùng).
- Phân tích Grad-CAM theo **lớp con** (vd: FAKE từ Stable Diffusion vs từ GAN) — yêu cầu dataset có nhãn nguồn.
- Đo độ tin cậy của Grad-CAM bằng **deletion / insertion metric**.

### 6.4 Tham khảo
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Bird & Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, 2023."""))

nb.cells = cells
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH} with {len(cells)} cells")
