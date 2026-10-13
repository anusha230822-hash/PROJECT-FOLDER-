import os
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from PIL import Image as PILImage
from .models import SatelliteImage, DatasetRecord

ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter password'}),
        min_length=6
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm password'})
    )

    class Meta:
        model = User
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email address'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            raise ValidationError("Passwords do not match. Please re-enter.")
        return cleaned_data


class UserLoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )


class SatelliteImageUploadForm(forms.ModelForm):
    class Meta:
        model = SatelliteImage
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={'class': 'form-control', 'id': 'image-upload-input', 'accept': '.jpg,.jpeg,.png,.tif,.tiff'})
        }

    def clean_image(self):
        img_file = self.cleaned_data.get('image')
        if not img_file:
            raise ValidationError("No image file provided.")

        ext = os.path.splitext(img_file.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '{ext}'. Please upload a JPG, JPEG, PNG, or TIFF image."
            )

        if img_file.size > MAX_FILE_SIZE_BYTES:
            raise ValidationError("Image file size exceeds the 10 MB limit.")

        # Validate that the file is actually an openable image
        try:
            pil_img = PILImage.open(img_file)
            pil_img.verify()
            # reset pointer after verify
            img_file.seek(0)
        except Exception as e:
            raise ValidationError(f"The uploaded file is corrupted or not a valid image: {str(e)}")

        return img_file


class DatasetRecordForm(forms.ModelForm):
    CLASS_CHOICES = (
        ('Forest', 'Forest'),
        ('Water', 'Water'),
        ('Agriculture', 'Agriculture'),
        ('Urban', 'Urban'),
        ('Barren', 'Barren'),
    )
    class_name = forms.ChoiceField(
        choices=CLASS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = DatasetRecord
        fields = ['image', 'class_name', 'dataset_type']
        widgets = {
            'image': forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': '.jpg,.jpeg,.png,.tif,.tiff'}),
            'dataset_type': forms.Select(attrs={'class': 'form-select'}),
        }
