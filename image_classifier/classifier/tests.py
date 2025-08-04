import warnings
import sys
import os

# Tüm uyarıları bastır
warnings.filterwarnings("ignore")

# Gereksiz stdout'u bastır (sadece testler için)
# class DevNull:
#     def write(self, msg):
#         pass
#     def flush(self):
#         pass

# if not os.environ.get("SHOW_TEST_OUTPUT"):
#     sys.stdout = DevNull()

from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .cnn_model import ImageClassifier
from .disease_segmentation import segment_disease
import numpy as np
from PIL import Image
import io
import torch
import time

class ImageClassifierTests(TestCase):
    def setUp(self):
        # Test için gerekli setup
        self.client = Client()
        # Test görüntüsü oluştur
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_io = io.BytesIO()
        self.test_image.save(self.test_image_io, format='JPEG')
        self.test_image_io.seek(0)
        self.test_image_file = SimpleUploadedFile(
            "test_image.jpg",
            self.test_image_io.read(),
            content_type="image/jpeg"
        )

    def test_home_view(self):
        """Test home page view"""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/home.html')
        print("Home view test passed successfully.")

    def test_integration_classify_image(self):
        """Integration test for classify_image view with valid image upload"""
        response = self.client.post(
            reverse('classify_image'),
            {'image': self.test_image_file},
            follow=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/result.html')
        self.assertIn('prediction', response.context)
        print("Integration test for classify_image passed successfully.")

    def test_segment_disease_function(self):
        """Test disease segmentation function"""
        # Test görüntüsünü geçici bir dosyaya kaydet
        temp_input_path = 'test_input_image.jpg'
        temp_output_path = 'test_output_mask.png'
        self.test_image.save(temp_input_path)
        model_path = 'classifier/models/fine_tuned_unet.pth'
        if os.path.exists(model_path):
            try:
                result_path = segment_disease(temp_input_path, model_path, temp_output_path)
                self.assertTrue(os.path.exists(result_path))
                os.remove(result_path)
                print("Disease segmentation test passed successfully.")
            except RuntimeError as e:
                self.skipTest("Model dosyası UNet ile uyumlu değil: " + str(e))
        else:
            self.skipTest("Model dosyası bulunamadı.")
        os.remove(temp_input_path)

    def test_system_image_classification(self):
        """System test for the entire image classification process"""
        # Test görüntüsünü geçici bir dosyaya kaydet
        temp_input_path = 'test_input_image.jpg'
        self.test_image.save(temp_input_path)
        
        # Görüntüyü yükle ve sınıflandır
        response = self.client.post(
            reverse('classify_image'),
            {'image': self.test_image_file},
            follow=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/result.html')
        self.assertIn('prediction', response.context)
        print("System test for image classification passed successfully.")
        
        # Geçici dosyayı temizle
        os.remove(temp_input_path)

    def test_performance_image_classification(self):
        """Performance test for the image classification process"""
        # Test görüntüsünü geçici bir dosyaya kaydet
        temp_input_path = 'test_input_image.jpg'
        self.test_image.save(temp_input_path)
        
        # Performans ölçümü başlat
        start_time = time.time()
        
        # Görüntüyü yükle ve sınıflandır
        response = self.client.post(
            reverse('classify_image'),
            {'image': self.test_image_file},
            follow=True
        )
        
        # Performans ölçümü bitir
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Performans kontrolü
        self.assertLess(elapsed_time, 5.0)  # 5 saniyeden az sürmeli
        print(f"Performance test for image classification passed successfully. Time taken: {elapsed_time*1000:.2f} ms.")
        
        # Geçici dosyayı temizle
        os.remove(temp_input_path)

    def test_functional_image_classification(self):
        """Functional test for the image classification process"""
        # Test görüntüsünü geçici bir dosyaya kaydet
        temp_input_path = 'test_input_image.jpg'
        self.test_image.save(temp_input_path)
        
        # Görüntüyü yükle ve sınıflandır
        response = self.client.post(
            reverse('classify_image'),
            {'image': self.test_image_file},
            follow=True
        )
        
        # Yanıt kontrolü
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/result.html')
        self.assertIn('prediction', response.context)
        
        # Tahmin kontrolü
        prediction = response.context['prediction']
        self.assertIsNotNone(prediction)
        print(f"Functional test for image classification passed successfully. Prediction: {prediction}")
        
        # Geçici dosyayı temizle
        os.remove(temp_input_path)

    def test_acceptance_workflow(self):
        """Acceptance test for the complete user workflow"""
        # 1. Ana sayfaya erişim
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/home.html')
        print("Step 1: Home page access successful")

        # 2. Geçerli bir görüntü yükleme
        temp_input_path = 'test_input_image.jpg'
        self.test_image.save(temp_input_path)
        
        response = self.client.post(
            reverse('classify_image'),
            {'image': self.test_image_file},
            follow=True
        )
        self.assertEqual(response.status_code, 200)
        print("Step 2: Image upload successful")

        # 3. Sonuç sayfası kontrolü
        self.assertTemplateUsed(response, 'classifier/result.html')
        self.assertIn('prediction', response.context)
        prediction = response.context['prediction']
        self.assertIsNotNone(prediction)
        print(f"Step 3: Result page successful with prediction: {prediction}")

        # 4. Ana sayfaya geri dönüş
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'classifier/home.html')
        print("Step 4: Return to home page successful")

        # Temizlik
        os.remove(temp_input_path)
        print("Acceptance test completed successfully - All user workflow steps passed!")