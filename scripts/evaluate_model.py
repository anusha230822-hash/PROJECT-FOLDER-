#!/usr/bin/env python
"""
Model Evaluation Script for Satellite Image Classification.
Run via:
    python scripts/evaluate_model.py
"""

import os
import sys
from pathlib import Path

# Set up Django environment
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satellite_project.settings')

import django
django.setup()

from django.conf import settings
from classifier.ml_classifier import is_model_trained
from classifier.evaluation import evaluate_test_dataset
from classifier.utils import get_dataset_statistics


def main():
    print("=" * 60)
    print("SATELLITE IMAGE CLASSIFICATION - PERFORMANCE EVALUATOR")
    print("=" * 60)

    if not is_model_trained():
        print("[!] ERROR: Model is not trained yet.")
        print("Please train the model first by running:")
        print("  python scripts/train_model.py")
        sys.exit(1)

    test_dir = settings.DATASET_DIR / 'test'
    stats = get_dataset_statistics()
    print(f"Testing split samples found: {stats['total_test']}")
    for cls_name, count in stats['test'].items():
        print(f"  - {cls_name}: {count} images")

    if stats['total_test'] == 0:
        print("\n[!] NOTICE: Test dataset directory contains 0 images.")
        print("Please place unseen test satellite images into:")
        print("  dataset/test/Forest/")
        print("  dataset/test/Water/")
        print("  dataset/test/Agriculture/")
        print("  dataset/test/Urban/")
        print("  dataset/test/Barren/")
        print("Then rerun: python scripts/evaluate_model.py")
        sys.exit(1)

    print("\nLoading test dataset...")
    print("Running predictions...")
    print("Calculating metrics...\n")

    try:
        results = evaluate_test_dataset(test_dir=test_dir)
        print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
        print(f"Precision: {results['precision'] * 100:.2f}%")
        print(f"Recall: {results['recall'] * 100:.2f}%")
        print(f"F1 Score: {results['f1_score'] * 100:.2f}%")
        print("\nConfusion matrix generated.")
        print("Evaluation results saved to ml_models/evaluation_results.json")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
