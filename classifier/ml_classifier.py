"""
Machine Learning Module for Satellite Land-Cover Classification.

Primary Classifier: RandomForestClassifier.
Supports:
1. Standard Scikit-Learn RandomForestClassifier when available in environment.
2. Built-in PureRandomForestClassifier (NumPy-based) as an automatic, transparent
   fallback when scikit-learn is unavailable or blocked by system policies.
3. Model persistence (Joblib / Pickle) and metadata tracking.
4. DEMO MODE: If model is not trained yet, raises ModelNotTrainedError.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np

from django.conf import settings
from .preprocessing import preprocess_image
from .feature_extraction import extract_features, FEATURE_NAMES


class ModelNotTrainedError(Exception):
    """Raised when image classification is attempted before model training."""
    pass


# ----------------------------------------------------------------------
# Pure Python / NumPy Random Forest Implementation (Autonomous Fallback)
# ----------------------------------------------------------------------

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, probs=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probs = probs

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=2, n_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        self.rng = rng if rng is not None else np.random.RandomState(42)

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        self.n_features_to_select = (
            int(np.sqrt(X.shape[1])) if self.n_features is None else min(self.n_features, X.shape[1])
        )
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            return self._make_leaf(y)

        # Randomly select subset of features
        feat_indices = self.rng.choice(n_feats, self.n_features_to_select, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_indices)
        if best_feat is None:
            return self._make_leaf(y)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._make_leaf(y)

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_indices):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        current_impurity = self._gini(y)

        for feat in feat_indices:
            values = X[:, feat]
            thresholds = np.unique(values)
            if len(thresholds) > 15:
                # Sample percentiles for fast, high-quality split evaluation
                thresholds = np.percentile(values, np.linspace(10, 90, 9))

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                p = np.sum(left_mask) / len(y)
                gain = current_impurity - (p * self._gini(y[left_mask]) + (1 - p) * self._gini(y[right_mask]))

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _make_leaf(self, y):
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / max(1, len(y))
        majority_class = int(np.argmax(counts))
        return TreeNode(value=majority_class, probs=probs)

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value, node.probs


class PureRandomForestClassifier:
    """
    Stand-alone, pure NumPy multi-class Random Forest classifier.
    Exposes identical interface to scikit-learn (fit, predict, predict_proba, classes_).
    """
    def __init__(self, n_estimators=50, max_depth=12, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        rng = np.random.RandomState(self.random_state)

        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                rng=np.random.RandomState(rng.randint(0, 100000))
            )
            tree.fit(X_boot, y_boot, n_classes=n_classes)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        all_tree_probs = []

        for tree in self.trees:
            tree_probs = [tree.predict_one(row)[1] for row in X]
            all_tree_probs.append(tree_probs)

        # Average probabilities across trees
        avg_probs = np.mean(all_tree_probs, axis=0)
        return avg_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class SimpleLabelEncoder:
    """Fallback label encoder if scikit-learn is not available."""
    def __init__(self):
        self.classes_ = None

    def fit_transform(self, y):
        self.classes_ = np.unique(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[val] for val in y], dtype=np.int64)

    def inverse_transform(self, y_indices):
        return np.array([self.classes_[i] for i in y_indices])


# ----------------------------------------------------------------------
# Paths and Core Model Functions
# ----------------------------------------------------------------------

def get_models_dir():
    if hasattr(settings, 'ML_MODELS_DIR'):
        return Path(settings.ML_MODELS_DIR)
    base = Path(__file__).resolve().parent.parent
    return base / 'ml_models'


def get_model_paths():
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        'model': models_dir / 'random_forest.pkl',
        'encoder': models_dir / 'label_encoder.pkl',
        'metadata': models_dir / 'model_metadata.json',
        'evaluation': models_dir / 'evaluation_results.json',
    }


def is_model_trained():
    paths = get_model_paths()
    return paths['model'].exists() and paths['encoder'].exists()


def load_model():
    """
    Loads trained model and encoder.
    Attempts joblib first; falls back to standard pickle.
    """
    if not is_model_trained():
        raise ModelNotTrainedError(
            "Model is not trained yet. Please add the dataset and run the training script."
        )

    paths = get_model_paths()

    try:
        import joblib
        model = joblib.load(paths['model'])
        label_encoder = joblib.load(paths['encoder'])
    except Exception:
        with open(paths['model'], 'rb') as f:
            model = pickle.load(f)
        with open(paths['encoder'], 'rb') as f:
            label_encoder = pickle.load(f)

    metadata = {}
    if paths['metadata'].exists():
        try:
            with open(paths['metadata'], 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return model, label_encoder, metadata


def get_prediction_confidence(model, feature_vector):
    feature_vector_2d = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(feature_vector_2d)[0]
        confidence = float(np.max(probabilities))
        classes = getattr(model, 'classes_', range(len(probabilities)))
        prob_dict = {str(c): float(round(p, 4)) for c, p in zip(classes, probabilities)}
        return confidence, prob_dict
    return 1.0, {}


def predict_features(feature_vector):
    model, label_encoder, metadata = load_model()

    feature_vector_2d = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    raw_pred = model.predict(feature_vector_2d)[0]

    if hasattr(label_encoder, 'inverse_transform'):
        predicted_class = label_encoder.inverse_transform([raw_pred])[0]
    else:
        predicted_class = str(raw_pred)

    confidence, raw_probs = get_prediction_confidence(model, feature_vector)

    named_probs = {}
    if hasattr(label_encoder, 'classes_'):
        for idx, cls_name in enumerate(label_encoder.classes_):
            named_probs[str(cls_name)] = raw_probs.get(idx, raw_probs.get(str(idx), 0.0))
    else:
        named_probs = raw_probs

    return str(predicted_class), float(confidence), named_probs


def predict_image(image_input):
    processed_rgb, _ = preprocess_image(image_input)
    feature_vector, feature_dict = extract_features(processed_rgb)
    predicted_class, confidence, probabilities = predict_features(feature_vector)
    return predicted_class, confidence, probabilities, feature_dict, processed_rgb


def train_model(train_dir=None, n_estimators=60, random_state=42):
    """
    Trains a Random Forest classifier on dataset/train/.
    Uses Scikit-Learn if operational; otherwise seamlessly uses PureRandomForestClassifier.
    """
    if train_dir is None:
        if hasattr(settings, 'DATASET_DIR'):
            train_dir = settings.DATASET_DIR / 'train'
        else:
            train_dir = Path(__file__).resolve().parent.parent / 'dataset' / 'train'

    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset directory not found at: {train_path}")

    class_folders = [d for d in train_path.iterdir() if d.is_dir()]
    if not class_folders:
        raise ValueError(f"No class folders found inside: {train_path}")

    features_list = []
    labels_list = []
    image_count = 0

    print(f"Reading training dataset from {train_path}...")
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        print(f"  Class '{class_name}': {len(images)} images found.")

        for img_path in images:
            try:
                processed_rgb, _ = preprocess_image(img_path)
                feat_vec, _ = extract_features(processed_rgb)
                features_list.append(feat_vec)
                labels_list.append(class_name)
                image_count += 1
            except Exception as e:
                print(f"  Warning: Skipping {img_path.name} due to error: {e}")

    if image_count == 0:
        raise ValueError("No valid images found to train the model.")

    X = np.array(features_list, dtype=np.float32)
    y_raw = np.array(labels_list)

    # Attempt scikit-learn; fall back to PureRandomForestClassifier if blocked
    use_sklearn = False
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        clf.fit(X, y)
        use_sklearn = True
        print(f"Trained using Scikit-Learn RandomForestClassifier.")
    except Exception as exc:
        print(f"Note: Using PureRandomForestClassifier (NumPy Engine). Reason: {exc}")
        label_encoder = SimpleLabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        clf = PureRandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X, y)

    paths = get_model_paths()

    # Save artifacts using pickle or joblib
    try:
        import joblib
        joblib.dump(clf, paths['model'])
        joblib.dump(label_encoder, paths['encoder'])
    except Exception:
        with open(paths['model'], 'wb') as f:
            pickle.dump(clf, f)
        with open(paths['encoder'], 'wb') as f:
            pickle.dump(label_encoder, f)

    metadata = {
        'model_type': 'Scikit-Learn RandomForestClassifier' if use_sklearn else 'PureRandomForestClassifier (NumPy)',
        'n_estimators': n_estimators,
        'features': FEATURE_NAMES,
        'classes': list(label_encoder.classes_),
        'total_samples': int(X.shape[0]),
        'feature_count': int(X.shape[1]),
    }

    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Model training completed successfully!")
    print(f"Model saved to: {paths['model']}")
    return clf, label_encoder, metadata
