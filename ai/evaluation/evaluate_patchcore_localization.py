anomaly_maps = torch.cat(
    all_anomaly_maps,
    dim=0,
)

masks = torch.stack(
    all_masks,
    dim=0,
)
# ============================================================
# Pixel-level threshold
# ============================================================

threshold = 0.117003

print()
print("=" * 60)
print("Pixel Localization Evaluation")
print("=" * 60)

print(
    f"Pixel threshold: {threshold:.6f}"
)


# ============================================================
# Binary predictions
# ============================================================

predictions = (
    anomaly_maps >= threshold
).float()


ground_truth = (
    masks > 0.5
).float()


# ============================================================
# Flatten
# ============================================================

predictions = predictions.reshape(-1)
ground_truth = ground_truth.reshape(-1)


# ============================================================
# Confusion components
# ============================================================

tp = (
    (predictions == 1) &
    (ground_truth == 1)
).sum().item()

tn = (
    (predictions == 0) &
    (ground_truth == 0)
).sum().item()

fp = (
    (predictions == 1) &
    (ground_truth == 0)
).sum().item()

fn = (
    (predictions == 0) &
    (ground_truth == 1)
).sum().item()


# ============================================================
# Metrics
# ============================================================

epsilon = 1e-8

precision = (
    tp / (tp + fp + epsilon)
)

recall = (
    tp / (tp + fn + epsilon)
)

f1 = (
    2 * precision * recall
    / (precision + recall + epsilon)
)

iou = (
    tp / (tp + fp + fn + epsilon)
)

dice = (
    2 * tp
    / (2 * tp + fp + fn + epsilon)
)


# ============================================================
# Results
# ============================================================

print()
print("Pixel Confusion Matrix")
print("-----------------------")

print(
    f"TP: {tp:,}"
)

print(
    f"TN: {tn:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)


print()
print("Pixel Metrics")
print("-----------------------")

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"IoU       : {iou:.4f}"
)

print(
    f"Dice      : {dice:.4f}"
)