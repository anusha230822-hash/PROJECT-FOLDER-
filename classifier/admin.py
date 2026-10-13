from django.contrib import admin
from .models import UserProfile, SatelliteImage, ClassificationResult, DatasetRecord, OntologyConcept

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'user_email')
    list_filter = ('role',)
    search_fields = ('user__username', 'user__email')

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'Email'


@admin.register(SatelliteImage)
class SatelliteImageAdmin(admin.ModelAdmin):
    list_display = ('image_name', 'user', 'processing_status', 'uploaded_at')
    list_filter = ('processing_status', 'uploaded_at')
    search_fields = ('image_name', 'user__username')
    readonly_fields = ('uploaded_at',)


@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = ('satellite_image', 'predicted_class', 'confidence_display', 'ontology_class', 'ontology_parent', 'created_at')
    list_filter = ('predicted_class', 'ontology_parent', 'created_at')
    search_fields = ('predicted_class', 'satellite_image__image_name', 'ontology_class')
    readonly_fields = ('created_at',)

    def confidence_display(self, obj):
        return f"{obj.confidence * 100:.1f}%"
    confidence_display.short_description = 'Confidence'


@admin.register(DatasetRecord)
class DatasetRecordAdmin(admin.ModelAdmin):
    list_display = ('class_name', 'dataset_type', 'uploaded_at')
    list_filter = ('class_name', 'dataset_type', 'uploaded_at')
    search_fields = ('class_name',)
    readonly_fields = ('uploaded_at',)


@admin.register(OntologyConcept)
class OntologyConceptAdmin(admin.ModelAdmin):
    list_display = ('concept_name', 'parent_concept', 'relationship', 'description')
    list_filter = ('relationship', 'parent_concept')
    search_fields = ('concept_name', 'parent_concept', 'description')
