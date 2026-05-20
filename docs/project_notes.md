# Project Notes

## Current Experimental Protocol

Date: 2026-05-16

The previous experiment results were removed because the test set was being used for early stopping. The project now uses a stricter protocol:

```text
train      -> learn model weights
validation -> select best checkpoint and apply early stopping
test       -> final evaluation only, after the best validation checkpoint is chosen
```

## Dataset Split

Original dataset:

```text
dataset/train/FAKE: 50000 images
dataset/train/REAL: 50000 images
dataset/test/FAKE: 10000 images
dataset/test/REAL: 10000 images
```

Current working split:

```text
training split:   80% of dataset/train, stratified by class
validation split: 20% of dataset/train, stratified by class
test split:       dataset/test, untouched until final evaluation
```

With the current `val_ratio = 0.2`, this means:

```text
training:   80000 images
validation: 20000 images
test:       20000 images
```

## Metrics

Each epoch tracks the following metrics for both train and validation:

```text
loss
accuracy
precision
recall
f1-score
```

The final test evaluation also reports:

```text
loss
accuracy
precision
recall
f1-score
```

## Early Stopping

Early stopping is based only on validation loss:

```text
monitor: validation loss
patience: 5 epochs
max_epochs: 50
```

The test set is not used for early stopping.

## Model Scripts

SimpleCNN:

```bash
python scripts/train_baseline.py
```

ResNet18:

```bash
python scripts/train_resnet18.py
```

EfficientNet-B0:

```bash
python scripts/train_efficientnet_b0.py
```

DenseNet121:

```bash
python scripts/train_densenet121.py
```

## Output Files

Checkpoints are saved to:

```text
outputs/checkpoints/
```

Metrics are saved to:

```text
outputs/metrics/
```

## Important Rule

Only final test metrics should be used for model comparison. Validation metrics are for model selection and early stopping.

## Completed Runs

All runs below use the corrected protocol:

```text
train -> validation early stopping -> final test evaluation
```

### SimpleCNN

Best validation checkpoint:

```text
best_epoch: 9
validation_loss: 0.1502
```

Final test metrics:

```text
test_loss: 0.1503
test_accuracy: 94.265%
test_precision: 95.312%
test_recall: 93.110%
test_f1: 94.198%
```

Output files:

```text
outputs/checkpoints/simple_cnn_best_model.pt
outputs/metrics/simple_cnn_metrics.json
```

### ResNet18

Best validation checkpoint:

```text
best_epoch: 9
validation_loss: 0.1369
```

Final test metrics:

```text
test_loss: 0.1371
test_accuracy: 94.800%
test_precision: 94.190%
test_recall: 95.490%
test_f1: 94.836%
```

Output files:

```text
outputs/checkpoints/resnet18_best_model.pt
outputs/metrics/resnet18_metrics.json
```

### EfficientNet-B0

Best validation checkpoint:

```text
best_epoch: 9
validation_loss: 0.1347
```

Final test metrics:

```text
test_loss: 0.1299
test_accuracy: 94.955%
test_precision: 95.005%
test_recall: 94.900%
test_f1: 94.952%
```

Output files:

```text
outputs/checkpoints/efficientnet_b0_best_model.pt
outputs/metrics/efficientnet_b0_metrics.json
```

### DenseNet121

Best validation checkpoint:

```text
best_epoch: 10
validation_loss: 0.1405
```

Final test metrics:

```text
test_loss: 0.1365
test_accuracy: 95.120%
test_precision: 94.401%
test_recall: 95.930%
test_f1: 95.159%
```

Output files:

```text
outputs/checkpoints/densenet121_best_model.pt
outputs/metrics/densenet121_metrics.json
```

## Model Comparison

Comparison uses final test metrics only:

| Model | Best Epoch | Validation Loss | Test Loss | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleCNN | 9 | 0.1502 | 0.1503 | 94.265% | 95.312% | 93.110% | 94.198% |
| ResNet18 | 9 | 0.1369 | 0.1371 | 94.800% | 94.190% | 95.490% | 94.836% |
| EfficientNet-B0 | 9 | 0.1347 | 0.1299 | 94.955% | 95.005% | 94.900% | 94.952% |
| DenseNet121 | 10 | 0.1405 | 0.1365 | 95.120% | 94.401% | 95.930% | 95.159% |

Current best model by final test F1-score:

```text
DenseNet121
```

Current best model by final test accuracy:

```text
DenseNet121
```

## Final Model Selection

Selected model for the next stage:

```text
DenseNet121Binary
```

Selected checkpoint:

```text
outputs/checkpoints/densenet121_best_model.pt
```

Reason:

DenseNet121 has the strongest final test performance among the completed models under the corrected train/validation/test protocol.

Selection metrics:

```text
test_accuracy: 95.120%
test_f1: 95.159%
test_loss: 0.1365
```

Next stage:

```text
Use DenseNet121 for Grad-CAM explainability.
```

## Gradio Demo

Demo file:

```text
app.py
```

Purpose:

```text
Provide a simple web interface for uploading an image, predicting REAL vs FAKE,
and visualizing the model attention with Grad-CAM.
```

Model used in the demo:

```text
DenseNet121Binary
```

Checkpoint:

```text
outputs/checkpoints/densenet121_best_model.pt
```

Demo outputs:

```text
1. REAL/FAKE prediction probabilities
2. Text summary with predicted label and confidence
3. Grad-CAM heatmap
4. Grad-CAM overlay on the input image
```

Run command:

```bash
python app.py
```

Local URL:

```text
Gradio prints the local URL after launch.
This app automatically picks an available port from 7860 to 8059.
You can force a specific port with GRADIO_SERVER_PORT if needed.
```
