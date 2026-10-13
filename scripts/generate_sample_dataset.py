#!/usr/bin/env python
"""
Generate Synthetic Satellite Demonstration Dataset.
Generates realistic multispectral patterns with representative color, texture, and edge signatures:
- Forest: High green channel, natural canopy noise, organic texture
- Water: Dominant blue/cyan, low edge density, smooth surface
- Agriculture: Geometric field bands, light greens and golden yellows
- Urban: High edge density, street grids, concrete grays and roofs
- Barren: High red/yellow tones, low moisture, dry sand and rocky textures

Run via:
    python scripts/generate_sample_dataset.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

CLASSES = ['Forest', 'Water', 'Agriculture', 'Urban', 'Barren']


def generate_forest_image(rng, size=(128, 128)):
    # Base deep green canopy
    base = np.zeros((*size, 3), dtype=np.uint8)
    base[:, :, 0] = rng.randint(20, 50)     # Low Red
    base[:, :, 1] = rng.randint(110, 160)   # High Green
    base[:, :, 2] = rng.randint(30, 60)     # Low Blue

    # Add canopy leaf texture noise
    noise = rng.normal(0, 18, (*size, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_water_image(rng, size=(128, 128)):
    # Deep blue / aquatic cyan
    base = np.zeros((*size, 3), dtype=np.uint8)
    base[:, :, 0] = rng.randint(10, 35)     # Very low Red
    base[:, :, 1] = rng.randint(60, 110)    # Moderate Green
    base[:, :, 2] = rng.randint(160, 220)   # Dominant Blue

    # Gentle wave ripples (smooth)
    noise = rng.normal(0, 6, (*size, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_agriculture_image(rng, size=(128, 128)):
    # Geometric croplands
    img = Image.new('RGB', size, (70, 140, 60))
    draw = ImageDraw.Draw(img)

    # Draw agricultural field plots (horizontal or vertical strips)
    strip_w = rng.randint(15, 30)
    for x in range(0, size[0], strip_w):
        r = rng.randint(80, 170)
        g = rng.randint(140, 210)
        b = rng.randint(40, 90)
        draw.rectangle([x, 0, x + strip_w, size[1]], fill=(r, g, b))

    # Add mild sensor noise
    arr = np.array(img).astype(np.int16) + rng.normal(0, 10, (*size, 3)).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_urban_image(rng, size=(128, 128)):
    # Built-up gray / asphalt background
    img = Image.new('RGB', size, (130, 130, 135))
    draw = ImageDraw.Draw(img)

    # Street network (dark lines)
    for _ in range(rng.randint(3, 6)):
        pos = rng.randint(10, 118)
        draw.line([(0, pos), (128, pos)], fill=(50, 50, 50), width=rng.randint(2, 4))
        draw.line([(pos, 0), (pos, 128)], fill=(50, 50, 50), width=rng.randint(2, 4))

    # Building rooftops (rectangles)
    for _ in range(rng.randint(8, 15)):
        bx = rng.randint(5, 105)
        by = rng.randint(5, 105)
        bw = rng.randint(10, 22)
        bh = rng.randint(10, 22)
        roof_colors = [
            (180, 70, 60),    # Brick red
            (210, 210, 215),  # White concrete
            (90, 95, 100),    # Dark asphalt
            (150, 160, 170),  # Metallic
        ]
        roof_color = roof_colors[rng.randint(len(roof_colors))]
        draw.rectangle([bx, by, bx + bw, by + bh], fill=roof_color)

    arr = np.array(img).astype(np.int16) + rng.normal(0, 12, (*size, 3)).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_barren_image(rng, size=(128, 128)):
    # Dry earth / desert tan
    base = np.zeros((*size, 3), dtype=np.uint8)
    base[:, :, 0] = rng.randint(170, 215)   # High Red
    base[:, :, 1] = rng.randint(140, 180)   # Moderate Green
    base[:, :, 2] = rng.randint(95, 130)    # Low Blue

    # Rough rocky gravel texture
    noise = rng.normal(0, 15, (*size, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


GENERATORS = {
    'Forest': generate_forest_image,
    'Water': generate_water_image,
    'Agriculture': generate_agriculture_image,
    'Urban': generate_urban_image,
    'Barren': generate_barren_image,
}


def main():
    print("=" * 60)
    print("SYNTHETIC SATELLITE DEMONSTRATION DATASET GENERATOR")
    print("=" * 60)

    dataset_root = BASE_DIR / 'dataset'
    train_dir = dataset_root / 'train'
    test_dir = dataset_root / 'test'

    n_train = 15  # 15 samples per class = 75 training samples
    n_test = 5    # 5 samples per class = 25 test samples

    rng = np.random.RandomState(42)

    total_created = 0
    for cls in CLASSES:
        gen = GENERATORS[cls]

        # Training set
        cls_train_dir = train_dir / cls
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, n_train + 1):
            img = gen(rng)
            out_path = cls_train_dir / f"{cls.lower()}_train_{i:03d}.png"
            img.save(out_path)
            total_created += 1

        # Testing set
        cls_test_dir = test_dir / cls
        cls_test_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, n_test + 1):
            img = gen(rng)
            out_path = cls_test_dir / f"{cls.lower()}_test_{i:03d}.png"
            img.save(out_path)
            total_created += 1

        print(f"Generated {cls}: {n_train} train images, {n_test} test images.")

    print(f"\n[SUCCESS] Generated {total_created} sample satellite images in {dataset_root}")
    print("Next steps:")
    print("  1. Train model:   python scripts/train_model.py")
    print("  2. Evaluate:      python scripts/evaluate_model.py")
    print("  3. Run server:    python manage.py runserver")


if __name__ == '__main__':
    main()
