// Main JavaScript for Satellite Image Classification System
document.addEventListener('DOMContentLoaded', function () {
    // 1. Image Upload Live Preview
    const uploadInput = document.getElementById('image-upload-input');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImage = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');

    if (uploadInput && previewImage) {
        uploadInput.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (event) {
                    previewImage.src = event.target.result;
                    previewImage.classList.remove('d-none');
                    if (uploadPrompt) {
                        uploadPrompt.classList.add('d-none');
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // 2. Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 6000);
    });
});
