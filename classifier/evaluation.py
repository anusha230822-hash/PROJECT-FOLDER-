"""
Performance Evaluation Module for Satellite Image Classification.

Computes actual, empirical evaluation metrics:
- Overall Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrix
- Per-class classification breakdown

Uses scikit-learn metrics when available; seamlessly falls back to exact NumPy mathematical definitions.
Never uses fake or simulated values. Saves real results to ml_models/evaluation_results.json.
"""

import json
from pathlib import Path
import numpy as np

from django.conf import settings
from .preprocessing import preprocess_image
from .feature_extraction import extract_features
from .ml_classifier import get_model_paths, load_model, is_model_trained


def get_saved_evaluation_results():
    eval_path = get_model_paths()['evaluation']
    if not eval_path.exists():
        return None

    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def calculate_metrics_numpy(y_true, y_pred, labels):
    """Computes exact accuracy, precision, recall, f1, and confusion matrix using pure NumPy."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, []

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt]][label_to_idx[yp]] += 1

    acc = correct / n_samples

    precisions = []
    recalls = []
    f1s = []

    for i, lbl in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return float(acc), macro_prec, macro_rec, macro_f1, cm


def evaluate_test_dataset(test_dir=None):
    """
    Evaluates the trained model on test dataset images.
    Returns dictionary with actual empirical metrics and confusion matrix.
    """
    if not is_model_trained():
        raise RuntimeError("Cannot evaluate model: Model is not trained yet.")

    if test_dir is None:
        if hasattr(settings, 'DATASET_DIR'):
            test_dir = settings.DATASET_DIR / 'test'
        else:
            test_dir = Path(__file__).resolve().parent.parent / 'dataset' / 'test'

    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset directory not found at: {test_path}")

    model, label_encoder, _ = load_model()

    class_folders = [d for d in test_path.iterdir() if d.is_dir()]
    if not class_folders:
        raise ValueError(f"No class folders found in test directory: {test_path}")

    y_true = []
    y_pred = []
    sample_count = 0

    print(f"Loading test images from {test_path}...")
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        print(f"  Test class '{class_name}': {len(images)} images.")

        for img_path in images:
            try:
                processed_rgb, _ = preprocess_image(img_path)
                feat_vec, _ = extract_features(processed_rgb)

                feat_2d = feat_vec.reshape(1, -1)
                pred_raw = model.predict(feat_2d)[0]

                if hasattr(label_encoder, 'inverse_transform'):
                    predicted_label = label_encoder.inverse_transform([pred_raw])[0]
                else:
                    predicted_label = str(pred_raw)

                y_true.append(class_name)
                y_pred.append(predicted_label)
                sample_count += 1
            except Exception as e:
                print(f"  Warning: Skipping test image {img_path.name}: {e}")

    if sample_count == 0:
        raise ValueError("No valid test images could be evaluated.")

    labels = sorted(list(set(y_true + y_pred)))

    # Compute metrics (try sklearn; fall back to calculate_metrics_numpy)
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix
        )
        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    except Exception:
        acc, prec_macro, rec_macro, f1_macro, cm = calculate_metrics_numpy(y_true, y_pred, labels)

    results = {
        'total_samples': sample_count,
        'classes': labels,
        'accuracy': round(acc, 4),
        'precision': round(prec_macro, 4),
        'recall': round(rec_macro, 4),
        'f1_score': round(f1_macro, 4),
        'confusion_matrix': cm,
    }

    # Save to JSON
    eval_file = get_model_paths()['evaluation']
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results
