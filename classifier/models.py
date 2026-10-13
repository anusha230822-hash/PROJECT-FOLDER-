from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# 1. User Profile Model
class UserProfile(models.Model):
    ROLE_CHOICES = (
        ('User', 'User'),
        ('Admin', 'Admin'),
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='User')

    def __str__(self):
        return f"{self.user.username} ({self.role})"


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        role = 'Admin' if instance.is_superuser else 'User'
        UserProfile.objects.create(user=instance, role=role)
    else:
        if hasattr(instance, 'profile'):
            if instance.is_superuser and instance.profile.role != 'Admin':
                instance.profile.role = 'Admin'
                instance.profile.save()


# 2. Satellite Image Model
class SatelliteImage(models.Model):
    STATUS_CHOICES = (
        ('Uploaded', 'Uploaded'),
        ('Processing', 'Processing'),
        ('Completed', 'Completed'),
        ('Failed', 'Failed'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='satellite_images')
    image = models.ImageField(upload_to='satellite_images/')
    image_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    processing_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Uploaded')
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.image_name} - {self.processing_status} ({self.user.username})"


# 3. Classification Result (Application-Level Knowledge Base)
class ClassificationResult(models.Model):
    satellite_image = models.OneToOneField(SatelliteImage, on_delete=models.CASCADE, related_name='result')
    predicted_class = models.CharField(max_length=100, db_index=True)
    confidence = models.FloatField(help_text="Prediction confidence between 0.0 and 1.0")
    ontology_class = models.CharField(max_length=100)
    ontology_parent = models.CharField(max_length=100)
    ontology_root = models.CharField(max_length=100, default='LandCover')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    features_json = models.TextField(blank=True, null=True, help_text="Serialized JSON of extracted features")
    probabilities_json = models.TextField(blank=True, null=True, help_text="Serialized JSON of class probabilities")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.satellite_image.image_name}: {self.predicted_class} ({self.confidence*100:.1f}%)"

    @property
    def confidence_percentage(self):
        return round(self.confidence * 100, 2)


# 4. Dataset Record Model (Admin Dataset Management)
class DatasetRecord(models.Model):
    DATASET_TYPE_CHOICES = (
        ('Training', 'Training'),
        ('Testing', 'Testing'),
    )

    image = models.ImageField(upload_to='dataset_records/')
    class_name = models.CharField(max_length=100, db_index=True)
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPE_CHOICES, default='Training')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.class_name} [{self.dataset_type}]"


# 5. Ontology Concept Model (Database View & Admin Management)
class OntologyConcept(models.Model):
    concept_name = models.CharField(max_length=100, unique=True, db_index=True)
    parent_concept = models.CharField(max_length=100, blank=True, null=True)
    relationship = models.CharField(max_length=100, default='isA')
    description = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['concept_name']

    def __str__(self):
        return f"{self.concept_name} -({self.relationship})-> {self.parent_concept or 'None'}"
