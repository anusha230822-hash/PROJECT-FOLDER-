"""
Ontology Mapping Module.
Maps machine learning classification outputs (Forest, Water, Agriculture, Urban, Barren)
to formal semantic OWL ontology concepts and retrieves hierarchical taxonomies.
"""

from .ontology import get_concept, get_parent_class, get_relationships

# Canonical dictionary mapping ML class outputs to OWL concept class names
ML_TO_ONTOLOGY_CLASS_MAP = {
    'Forest': 'Forest',
    'Agriculture': 'Agriculture',
    'Water': 'WaterBody',
    'Urban': 'UrbanArea',
    'Barren': 'BarrenLand',
    'Barren Land': 'BarrenLand',
}


def map_prediction_to_ontology(predicted_class):
    """
    Given an ML predicted class name, maps it to the semantic ontology concept
    and extracts parent, root, and relational taxonomic structure.
    """
    clean_name = str(predicted_class).strip()
    concept_name = ML_TO_ONTOLOGY_CLASS_MAP.get(clean_name, clean_name)

    # Fetch concept from OWL file
    concept_obj = get_concept(concept_name)

    # Defaults in case concept was not in OWL file
    ontology_class = concept_name
    parent_class = 'LandCover'
    root_class = 'LandCover'
    description = ''

    if concept_obj is not None:
        ontology_class = concept_obj.name
        if hasattr(concept_obj, 'comment') and concept_obj.comment:
            description = str(concept_obj.comment[0])

        # Get parent from ontology
        parent_info = get_parent_class(concept_obj.name)
        if parent_info:
            parent_class = parent_info['name']

        # Traverse upwards to root
        rels = get_relationships(concept_obj.name)
        if rels:
            root_class = rels[-1]['target']
    else:
        # Fallback mappings for robust demonstration
        if clean_name in ['Forest', 'Agriculture']:
            parent_class = 'Vegetation'
        elif clean_name == 'Water':
            parent_class = 'LandCover'
        elif clean_name == 'Urban':
            parent_class = 'BuiltUpArea'
        elif clean_name in ['Barren', 'Barren Land']:
            parent_class = 'LandCover'

    # Build taxonomy breadcrumb path
    hierarchy = [root_class]
    if parent_class != root_class:
        hierarchy.append(parent_class)
    if ontology_class not in hierarchy:
        hierarchy.append(ontology_class)

    hierarchy_path = " > ".join(hierarchy)

    return {
        'predicted_class': clean_name,
        'ontology_class': ontology_class,
        'parent_class': parent_class,
        'root_class': root_class,
        'description': description,
        'hierarchy_path': hierarchy_path,
        'relationships': get_relationships(ontology_class) if concept_obj else []
    }
