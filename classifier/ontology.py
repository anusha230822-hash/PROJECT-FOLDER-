"""
OWL / Ontology Integration Module.
Integrates with Owlready2 to load and query ontology/satellite_ontology.owl.
Provides hierarchical traversal, search, and relationship extraction.
"""

import os
from pathlib import Path
import owlready2
from django.conf import settings

_LOADED_ONTO = None


def get_ontology_path():
    """Returns absolute path to the OWL ontology file."""
    if hasattr(settings, 'ONTOLOGY_FILE_PATH'):
        return str(settings.ONTOLOGY_FILE_PATH)
    base = Path(__file__).resolve().parent.parent
    return str(base / 'ontology' / 'satellite_ontology.owl')


def load_ontology(owl_path=None):
    """
    Loads or returns cached Owlready2 ontology instance.
    Handles errors gracefully with friendly messages.
    """
    global _LOADED_ONTO
    if _LOADED_ONTO is not None:
        return _LOADED_ONTO

    path = owl_path or get_ontology_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ontology file not found at: {path}")

    try:
        # Load using Owlready2
        onto = owlready2.get_ontology(f"file://{os.path.abspath(path)}").load()
        _LOADED_ONTO = onto
        return _LOADED_ONTO
    except Exception as e:
        raise RuntimeError(f"Error loading OWL ontology file: {str(e)}")


def get_concept(concept_name):
    """
    Retrieves an ontology class by name (case-insensitive lookup).
    Returns class object or None.
    """
    onto = load_ontology()
    for cls in onto.classes():
        if cls.name.lower() == concept_name.lower().replace(' ', ''):
            return cls
        # Check rdfs:label
        if hasattr(cls, 'label') and cls.label:
            for lbl in cls.label:
                if str(lbl).lower() == concept_name.lower():
                    return cls
    return None


def get_parent_class(concept_name):
    """
    Finds direct parent class in the ontology hierarchy.
    Returns (parent_name, parent_label) or None.
    """
    concept = get_concept(concept_name)
    if not concept:
        return None

    parents = [p for p in concept.is_a if hasattr(p, 'name') and p.name != 'Thing']
    if parents:
        parent = parents[0]
        label = parent.label[0] if (hasattr(parent, 'label') and parent.label) else parent.name
        return {
            'name': parent.name,
            'label': str(label)
        }
    return None


def get_children(concept_name):
    """
    Finds all direct subclasses of a concept.
    Returns list of dicts: [{'name': ..., 'label': ..., 'description': ...}]
    """
    concept = get_concept(concept_name)
    if not concept:
        return []

    onto = load_ontology()
    children = []
    for cls in onto.classes():
        parents = [p for p in cls.is_a if hasattr(p, 'name')]
        if concept in parents and cls != concept:
            lbl = cls.label[0] if (hasattr(cls, 'label') and cls.label) else cls.name
            desc = cls.comment[0] if (hasattr(cls, 'comment') and cls.comment) else ''
            children.append({
                'name': cls.name,
                'label': str(lbl),
                'description': str(desc)
            })
    return children


def get_relationships(concept_name):
    """
    Traverses complete taxonomy chain from concept up to LandCover.
    Returns list of relationships e.g. [('Forest', 'isA', 'Vegetation'), ('Vegetation', 'isA', 'LandCover')].
    """
    concept = get_concept(concept_name)
    if not concept:
        return []

    chain = []
    curr = concept
    while curr and hasattr(curr, 'is_a'):
        parents = [p for p in curr.is_a if hasattr(p, 'name') and p.name != 'Thing']
        if not parents:
            break
        p = parents[0]
        chain.append({
            'source': curr.name,
            'relationship': 'isA (SubClassOf)',
            'target': p.name,
            'source_label': str(curr.label[0]) if (hasattr(curr, 'label') and curr.label) else curr.name,
            'target_label': str(p.label[0]) if (hasattr(p, 'label') and p.label) else p.name,
        })
        curr = p
    return chain


def search_ontology(query):
    """
    Searches ontology concepts by name, label, or description.
    Returns list of matched concept details with ancestry and children.
    """
    if not query:
        return []

    onto = load_ontology()
    q = query.strip().lower().replace(' ', '')
    results = []

    for cls in onto.classes():
        name_match = q in cls.name.lower()
        lbl = str(cls.label[0]) if (hasattr(cls, 'label') and cls.label) else cls.name
        label_match = q in lbl.lower()
        desc = str(cls.comment[0]) if (hasattr(cls, 'comment') and cls.comment) else ''
        desc_match = q in desc.lower()

        if name_match or label_match or desc_match:
            parent = get_parent_class(cls.name)
            children = get_children(cls.name)
            rel = get_relationships(cls.name)
            results.append({
                'name': cls.name,
                'label': lbl,
                'description': desc,
                'parent': parent['name'] if parent else 'Root/None',
                'parent_label': parent['label'] if parent else 'Root/None',
                'children': children,
                'relationships': rel
            })

    return results


def get_all_concepts_tree():
    """
    Builds a nested dictionary representation of the entire ontology tree.
    Suitable for frontend visualization and ontology explorer.
    """
    onto = load_ontology()
    root_class = get_concept('LandCover')
    if not root_class:
        return {}

    def build_subtree(cls):
        lbl = str(cls.label[0]) if (hasattr(cls, 'label') and cls.label) else cls.name
        desc = str(cls.comment[0]) if (hasattr(cls, 'comment') and cls.comment) else ''
        subclasses = []
        for other in onto.classes():
            if other != cls and cls in other.is_a:
                subclasses.append(build_subtree(other))
        return {
            'name': cls.name,
            'label': lbl,
            'description': desc,
            'children': subclasses
        }

    return build_subtree(root_class)
