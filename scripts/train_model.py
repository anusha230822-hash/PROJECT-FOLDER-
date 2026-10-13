#!/usr/bin/env python
"""
Model Training Script for Satellite Image Classification.
Run via:
    python scripts/train_model.py
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
from classifier.ml_classifier import train_model, get_model_paths
from classifier.utils import get_dataset_statistics


def main():
    print("=" * 60)
    print("SATELLITE IMAGE CLASSIFICATION - RANDOM FOREST TRAINER")
    print("=" * 60)

    train_dir = settings.DATASET_DIR / 'train'
    stats = get_dataset_statistics()
    print(f"Dataset root: {settings.DATASET_DIR}")
    print(f"Training split samples found: {stats['total_train']}")
    for cls_name, count in stats['train'].items():
        print(f"  - {cls_name}: {count} images")

    if stats['total_train'] == 0:
        print("\n[!] NOTICE: Training dataset directory contains 0 images.")
        print("Please place satellite images into:")
        print("  dataset/train/Forest/")
        print("  dataset/train/Water/")
        print("  dataset/train/Agriculture/")
        print("  dataset/train/Urban/")
        print("  dataset/train/Barren/")
        print("Then rerun: python scripts/train_model.py")
        sys.exit(1)

    print("\nLoading dataset...")
    print("Preprocessing images...")
    print("Extracting features (15 numerical dimensions)...")
    print("Training Random Forest Classifier...")

    try:
        clf, encoder, metadata = train_model(train_dir=train_dir)
        paths = get_model_paths()
        print("\n[SUCCESS] Model training completed successfully!")
        print(f"Model saved to: {paths['model']}")
        print(f"Label encoder saved to: {paths['encoder']}")
        print(f"Metadata saved to: {paths['metadata']}")
        print(f"Total training samples used: {metadata['total_samples']}")
        print(f"Classes trained: {', '.join(metadata['classes'])}")
    except Exception as e:
        print(f"\n[ERROR] Model training could not be completed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
