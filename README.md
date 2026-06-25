# CIFAKE Image Classification with Explainable AI

Dự án này xây dựng pipeline phân loại ảnh **REAL** và **FAKE** trên dataset CIFAKE, sau đó dùng **Grad-CAM** để giải thích vùng ảnh mà model chú ý khi đưa ra dự đoán.

Project được làm theo hướng học từng bước:

```text
dataset exploration -> dataset/dataloader -> baseline CNN
-> deep CNN models -> model comparison -> Grad-CAM explainability
-> Gradio demo
```

## Mục Tiêu

- Kiểm tra và hiểu dataset CIFAKE.
- Tự xây dựng dataloader cho bài toán binary classification.
- Train và so sánh nhiều kiến trúc CNN.
- Chọn best model dựa trên final test metrics.
- Dùng Grad-CAM để giải thích model học/chú ý vào vùng nào.
- Làm demo web đơn giản bằng Gradio để upload ảnh và xem dự đoán.

## Dataset

Nguồn dữ liệu Kaggle:

[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Dataset được đặt trong thư mục:

```text
dataset/
```

Thư mục `dataset/` không nên commit lên GitHub vì dung lượng lớn. Khi clone project ở máy khác, hãy tải dataset từ Kaggle rồi giải nén theo đúng cấu trúc bên dưới.

Cấu trúc dữ liệu hiện tại:

```text
dataset/
├── train/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

Số lượng ảnh đã kiểm tra:

| Split | Class | Images |
|---|---:|---:|
| train | REAL | 50,000 |
| train | FAKE | 50,000 |
| test | REAL | 10,000 |
| test | FAKE | 10,000 |

Trong code train, tập `train` gốc được chia tiếp thành:

```text
80% train
20% validation
```

Tập `test` chỉ dùng một lần ở cuối để đánh giá model tốt nhất.

## Cấu Trúc Project

```text
.
├── app.py                         # Gradio demo
├── configs/                       # File cấu hình
├── dataset/                       # CIFAKE dataset
├── docs/
│   └── project_notes.md           # Ghi chú quá trình làm project
├── notebooks/
│   └── gradcam_analysis.ipynb     # Phân tích Grad-CAM
├── outputs/
│   ├── checkpoints/               # Model checkpoints
│   ├── metrics/                   # Metrics JSON
│   └── gradcam_batch/             # Heatmap/metadata Grad-CAM
├── references/                    # Ghi chú paper
├── scripts/                       # Script chạy từng bước
├── src/                           # Source code chính
│   ├── data/                      # Dataset, dataloader, transforms
│   ├── explainability/            # Grad-CAM, heatmap stats
│   ├── models/                    # CNN, ResNet, EfficientNet, DenseNet
│   ├── training/                  # Train/evaluate loop
│   └── utils/                     # Metrics helper
└── tests/                         # Unit tests
```

## Cài Đặt Môi Trường

Tạo và kích hoạt virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Nếu đã có `.venv`, chỉ cần kích hoạt:

```bash
source .venv/bin/activate
```

## Kiểm Tra Dataset

```bash
python scripts/dataset_exploration.py
```

Script này kiểm tra:

- số lượng ảnh theo split/class
- kích thước ảnh
- ảnh lỗi/corrupted image

## Train Models

Train SimpleCNN baseline:

```bash
python scripts/train_baseline.py
```

Train ResNet18:

```bash
python scripts/train_resnet18.py
```

Train EfficientNet-B0:

```bash
python scripts/train_efficientnet_b0.py
```

Train DenseNet121:

```bash
python scripts/train_densenet121.py
```

Train DeiT-Tiny:

```bash
python scripts/train_deit_tiny.py
```

Các script train đều dùng:

```text
max_epochs = 50
early_stopping_patience = 5
validation split = 20% từ train gốc
```

Metric theo dõi mỗi epoch:

```text
loss, accuracy, precision, recall, f1-score
```

## Model Comparison

Kết quả final test hiện tại:

| Model | Best Epoch | Validation Loss | Test Loss | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleCNN | 9 | 0.1502 | 0.1503 | 94.265% | 95.312% | 93.110% | 94.198% |
| ResNet18 | 9 | 0.1369 | 0.1371 | 94.800% | 94.190% | 95.490% | 94.836% |
| EfficientNet-B0 | 9 | 0.1347 | 0.1299 | 94.955% | 95.005% | 94.900% | 94.952% |
| DenseNet121 | 10 | 0.1405 | 0.1365 | 95.120% | 94.401% | 95.930% | 95.159% |

Best model hiện tại:

```text
DenseNet121Binary
```

Best checkpoint:

```text
outputs/checkpoints/densenet121_best_model.pt
```

## Evaluate Models

Evaluate SimpleCNN:

```bash
python scripts/evaluate_simple_cnn.py
```

Evaluate ResNet18:

```bash
python scripts/evaluate_resnet18.py
```

Evaluate EfficientNet-B0:

```bash
python scripts/evaluate_efficientnet_b0.py
```

Evaluate DeiT-Tiny:

```bash
python scripts/evaluate_deit_tiny.py
```

DenseNet121 final test metrics được lưu khi chạy:

```bash
python scripts/train_densenet121.py
```

DeiT-Tiny final test metrics được lưu khi chạy:

```bash
python scripts/train_deit_tiny.py
```

## Grad-CAM Explainability

Sinh Grad-CAM batch cho các model:

```bash
python scripts/generate_gradcam_batch.py --num 50
```

Kết quả được lưu tại:

```text
outputs/gradcam_batch/
```

Notebook phân tích:

```text
notebooks/gradcam_analysis.ipynb
```

Notebook này dùng để:

- xem Grad-CAM overlay
- so sánh attention giữa các model
- phân tích case đúng/sai
- đo coverage, entropy, peak value của heatmap

## Gradio Demo

Chạy demo:

```bash
source .venv/bin/activate
python app.py
```

Sau đó mở URL mà Gradio in ra trong terminal, thường có dạng:

```text
http://127.0.0.1:7860
```

Demo cho phép:

- upload ảnh
- dự đoán REAL/FAKE
- xem xác suất `P(REAL)` và `P(FAKE)`
- xem Grad-CAM heatmap
- xem Grad-CAM overlay trên ảnh đầu vào

Nếu muốn ép app chạy ở một port cụ thể:

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

Dừng demo:

```text
Ctrl + C
```

## Chạy Tests

```bash
python -m unittest discover tests
```

Các test hiện tại kiểm tra:

- dataset/dataloader
- metrics
- heatmap stats
- model forward pass
- helper functions của Gradio demo

## Ghi Chú Quan Trọng

- Dataset CIFAKE có ảnh kích thước nhỏ `32x32`, nên Grad-CAM nhìn sẽ khá thô.
- Early stopping dùng validation loss, không dùng test loss.
- Test set chỉ dùng để đánh giá cuối cùng sau khi chọn checkpoint tốt nhất.
- DenseNet121 đang là model tốt nhất trong các model đã train ở project này.
- Checkpoint và outputs có thể lớn, nên thường không commit trực tiếp lên git.

## Tài Liệu Tham Khảo

- Bird & Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, 2023.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
