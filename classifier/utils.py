"""
Utility functions for dataset inspection, formatting, and database initialization.
"""

from pathlib import Path
from django.conf import settings
from .models import OntologyConcept
from .ontology import get_all_concepts_tree, load_ontology

ALL_CLASSES = ['Forest', 'Water', 'Agriculture', 'Urban', 'Barren']


def get_dataset_statistics():
    """
    Scans the physical dataset directory and returns file counts.
    Returns:
    {
       'train': {'Forest': n, 'Water': n, ...},
       'test': {'Forest': n, 'Water': n, ...},
       'total_train': int,
       'total_test': int,
    }
    """
    base_dir = Path(settings.DATASET_DIR) if hasattr(settings, 'DATASET_DIR') else Path('dataset')
    train_dir = base_dir / 'train'
    test_dir = base_dir / 'test'

    stats = {
        'train': {},
        'test': {},
        'total_train': 0,
        'total_test': 0,
    }

    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    for cls in ALL_CLASSES:
        # Train count
        cls_train_path = train_dir / cls
        train_count = 0
        if cls_train_path.exists():
            train_count = sum(1 for f in cls_train_path.iterdir() if f.suffix.lower() in valid_exts)
        stats['train'][cls] = train_count
        stats['total_train'] += train_count

        # Test count
        cls_test_path = test_dir / cls
        test_count = 0
        if cls_test_path.exists():
            test_count = sum(1 for f in cls_test_path.iterdir() if f.suffix.lower() in valid_exts)
        stats['test'][cls] = test_count
        stats['total_test'] += test_count

    return stats


def populate_initial_ontology_concepts_in_db():
    """
    Synchronizes concepts from the OWL ontology file into the Django SQLite OntologyConcept table.
    Ensures admin and search queries have full database representations.
    """
    try:
        onto = load_ontology()
        for cls in onto.classes():
            name = cls.name
            desc = str(cls.comment[0]) if (hasattr(cls, 'comment') and cls.comment) else ''
            parents = [p.name for p in cls.is_a if hasattr(p, 'name') and p.name != 'Thing']
            parent_name = parents[0] if parents else None

            OntologyConcept.objects.get_or_create(
                concept_name=name,
                defaults={
                    'parent_concept': parent_name,
                    'relationship': 'isA (SubClassOf)',
                    'description': desc
                }
            )
    except Exception as e:
        print(f"Notice: Could not sync ontology concepts to DB: {e}")
