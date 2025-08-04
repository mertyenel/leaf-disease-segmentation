from django.core.management.base import BaseCommand
from django.conf import settings
import os
from classifier.models import LeafImage
from classifier.cnn_model import ImageClassifier

class Command(BaseCommand):
    help = 'Test klasöründeki görüntüleri veritabanına yükler'

    def handle(self, *args, **options):
        # Test klasörü yolunu al
        test_images_path = os.path.join(
            settings.BASE_DIR,
            'classifier', 'static', 'classifier', 'test_images'
        )

        # Klasörün varlığını kontrol et
        if not os.path.exists(test_images_path):
            self.stdout.write(self.style.ERROR(f'Test klasörü bulunamadı: {test_images_path}'))
            return

        # Sınıflandırıcı modelini yükle
        model_path = os.path.join(settings.BASE_DIR, 'classifier', 'models', 'model.pth')
        classifier = ImageClassifier(model_path)

        # Geçerli görüntü uzantıları
        valid_exts = ('.jpg', '.jpeg', '.png')
        
        # Klasördeki tüm görüntüleri bul
        images = [f for f in os.listdir(test_images_path) if f.lower().endswith(valid_exts)]
        
        if not images:
            self.stdout.write(self.style.WARNING('Test klasöründe görüntü bulunamadı.'))
            return

        # Her görüntü için
        for image_name in images:
            # Eğer görüntü zaten veritabanında varsa atla
            if LeafImage.objects.filter(image_name=image_name).exists():
                self.stdout.write(f'Görüntü zaten veritabanında: {image_name}')
                continue

            image_path = os.path.join(test_images_path, image_name)
            
            try:
                # Görüntüyü oku ve binary formata çevir
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                # Görüntüyü sınıflandır
                prediction = classifier.predict(image_path)
                
                # Tahmin sonucunu parçala
                type_, condition = prediction.split()
                
                # Veritabanına kaydet
                LeafImage.objects.create(
                    image_name=image_name,
                    image_data=image_data,
                    type=type_,
                    condition=condition
                )
                
                self.stdout.write(self.style.SUCCESS(f'Görüntü başarıyla yüklendi: {image_name}'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Görüntü yüklenirken hata oluştu {image_name}: {str(e)}'))

        self.stdout.write(self.style.SUCCESS('Tüm görüntüler işlendi.')) 