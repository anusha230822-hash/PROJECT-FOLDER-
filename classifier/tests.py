import io
from PIL import Image as PILImage
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile

from classifier.models import (
    SatelliteImage, ClassificationResult, UserProfile,
    OntologyConcept
)
from classifier.preprocessing import preprocess_image, resize_image, normalize_image
from classifier.feature_extraction import extract_features, FEATURE_NAMES
from classifier.ontology import load_ontology, get_concept, get_parent_class, search_ontology
from classifier.ontology_mapping import map_prediction_to_ontology


def create_test_image_file(name="test_sat.png", size=(100, 100), color=(34, 139, 34)):
    """Helper to generate an in-memory valid PNG test image."""
    image_io = io.BytesIO()
    image = PILImage.new('RGB', size, color=color)
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return SimpleUploadedFile(name, image_io.read(), content_type='image/png')


class AuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='teststudent', password='password123', email='student@test.com')

    def test_user_registration(self):
        response = self.client.post(reverse('register'), {
            'username': 'newuser',
            'email': 'newuser@test.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_user_login_and_logout(self):
        login_res = self.client.post(reverse('login'), {
            'username': 'teststudent',
            'password': 'password123'
        })
        self.assertEqual(login_res.status_code, 302)
        self.assertTrue('_auth_user_id' in self.client.session)

        logout_res = self.client.get(reverse('logout'))
        self.assertEqual(logout_res.status_code, 302)
        self.assertFalse('_auth_user_id' in self.client.session)


class PreprocessingAndFeatureExtractionTests(TestCase):
    def test_image_preprocessing(self):
        img_file = create_test_image_file(size=(200, 150), color=(0, 100, 200))
        processed_rgb, normalized_rgb = preprocess_image(img_file)

        # Check resized dimensions (128x128)
        self.assertEqual(processed_rgb.shape, (128, 128, 3))
        self.assertEqual(normalized_rgb.shape, (128, 128, 3))
        # Check normalized range
        self.assertLessEqual(normalized_rgb.max(), 1.0)
        self.assertGreaterEqual(normalized_rgb.min(), 0.0)

    def test_feature_extraction(self):
        img_file = create_test_image_file(size=(128, 128), color=(40, 160, 40))
        processed_rgb, _ = preprocess_image(img_file)
        vector, feat_dict = extract_features(processed_rgb)

        # Check feature count
        self.assertEqual(len(vector), 15)
        for name in FEATURE_NAMES:
            self.assertIn(name, feat_dict)
            self.assertIsInstance(feat_dict[name], (int, float))


class OntologyTests(TestCase):
    def test_ontology_loading(self):
        onto = load_ontology()
        self.assertIsNotNone(onto)
        class_names = [c.name for c in onto.classes()]
        self.assertIn('LandCover', class_names)
        self.assertIn('Vegetation', class_names)
        self.assertIn('Forest', class_names)
        self.assertIn('WaterBody', class_names)

    def test_ontology_parent_traversal(self):
        parent = get_parent_class('Forest')
        self.assertIsNotNone(parent)
        self.assertEqual(parent['name'], 'Vegetation')

    def test_ontology_mapping(self):
        for cls_name in ['Forest', 'Water', 'Agriculture', 'Urban', 'Barren']:
            res = map_prediction_to_ontology(cls_name)
            self.assertEqual(res['root_class'], 'LandCover')
            self.assertTrue(len(res['hierarchy_path']) > 0)
            self.assertEqual(res['predicted_class'], cls_name)

    def test_ontology_search(self):
        results = search_ontology('Vegetation')
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]['name'], 'Vegetation')


class ImageUploadAndResultTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='satuser', password='password123')
        self.client.login(username='satuser', password='password123')

    def test_valid_image_upload(self):
        img_file = create_test_image_file(name="forest_sample.png")
        response = self.client.post(reverse('upload'), {'image': img_file})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(SatelliteImage.objects.count(), 1)
        sat_img = SatelliteImage.objects.first()
        self.assertEqual(sat_img.user, self.user)

    def test_invalid_image_extension_upload(self):
        bad_file = SimpleUploadedFile("malicious.exe", b"binary content", content_type='application/octet-stream')
        response = self.client.post(reverse('upload'), {'image': bad_file})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(SatelliteImage.objects.count(), 0)
        self.assertTrue(response.context['form'].errors)

    def test_knowledge_base_record_creation(self):
        sat_img = SatelliteImage.objects.create(
            user=self.user,
            image=create_test_image_file(),
            image_name='test.png',
            processing_status='Completed'
        )
        res = ClassificationResult.objects.create(
            satellite_image=sat_img,
            predicted_class='Forest',
            confidence=0.92,
            ontology_class='Forest',
            ontology_parent='Vegetation',
            ontology_root='LandCover'
        )
        self.assertEqual(res.confidence_percentage, 92.0)
        self.assertEqual(sat_img.result.predicted_class, 'Forest')
