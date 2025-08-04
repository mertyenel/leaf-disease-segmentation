# classifier/views.py

import os
import random
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import cv2
import numpy as np
from django.http import HttpResponse
from .models import LeafImage

from .cnn_model import ImageClassifier
from .unet import segment_leaf  # segment fonksiyonunu unet.py'den al
from .disease_segmentation import segment_disease  # hastalık segmentasyonu için
from .overlay import create_overlay  # overlay görüntü oluşturma için

# Sınıflandırma modeli
MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier', 'models', 'model.pth')
classifier = ImageClassifier(MODEL_PATH)

# Segmentasyon modelleri
WHOLE_LEAF_MODEL = os.path.join(settings.BASE_DIR, 'classifier', 'models', 'whole_leaf.pth')
DISEASE_MODEL = os.path.join(settings.BASE_DIR, 'classifier', 'models', 'fine_tuned_unet.pth')

def get_image_from_db(image_name):
    """Veritabanından görüntüyü alır ve geçici bir dosya olarak kaydeder."""
    try:
        leaf_image = LeafImage.objects.get(image_name=image_name)
        # Geçici dosya oluştur
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', image_name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Görüntüyü geçici dosyaya kaydet
        with open(temp_path, 'wb') as f:
            f.write(leaf_image.image_data)
        
        return temp_path
    except LeafImage.DoesNotExist:
        return None

def home(request):
    return render(request, 'classifier/home.html')

def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        # Check if the file is a valid image
        if not image.content_type.startswith('image/'):
            return render(request, 'classifier/home.html', {'error': 'Lütfen geçerli bir görüntü dosyası yükleyin.'})
        
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        image_path = fs.path(filename)
        prediction = classifier.predict(image_path)
        
        # Eğer görüntü yaprak değilse, sadece sonucu göster
        if prediction == "Yaprak değil":
            return render(request, 'classifier/result.html', {
                'uploaded_file_url': uploaded_file_url,
                'prediction': prediction,
            })
        
        # Görüntü yaprak ise, veritabanına kaydet
        type_, condition = prediction.split()
        # Görüntüyü veritabanına kaydet
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        LeafImage.objects.create(
            image_name=filename,
            image_data=image_data,
            type=type_,
            condition=condition
        )

        return render(request, 'classifier/result.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': prediction,
        })
    return render(request, 'classifier/home.html')

def random_classify_image(request):
    if request.method == 'POST':
        # Veritabanından rastgele bir görüntü seç
        leaf_images = LeafImage.objects.all()
        if not leaf_images.exists():
            return render(request, 'classifier/result.html', {
                'uploaded_file_url': None,
                'prediction': "Veritabanında görüntü yok.",
            })

        leaf_image = random.choice(list(leaf_images))
        
        # Görüntüyü geçici dosyaya kaydet
        temp_path = get_image_from_db(leaf_image.image_name)
        if not temp_path:
            return render(request, 'classifier/result.html', {
                'uploaded_file_url': None,
                'prediction': "Görüntü bulunamadı.",
            })
        
        uploaded_file_url = f'/media/temp/{leaf_image.image_name}'
        
        return render(request, 'classifier/result.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': f"{leaf_image.type} {leaf_image.condition}",
        })
    return render(request, 'classifier/home.html')

def calculate_disease_percentage(leaf_mask_path, disease_mask_path):
    """
    Yaprak ve hastalık maskelerindeki beyaz pikselleri sayarak hastalık yüzdesini hesaplar.
    
    Args:
        leaf_mask_path (str): Yaprak maskesinin yolu
        disease_mask_path (str): Hastalık maskesinin yolu
    
    Returns:
        tuple: (yaprak piksel sayısı, hastalık piksel sayısı, hastalık yüzdesi)
    """
    # Maskeleri oku
    leaf_mask = cv2.imread(leaf_mask_path, cv2.IMREAD_GRAYSCALE)
    disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Hastalık maskesini yaprak maskesi boyutuna getir
    disease_mask = cv2.resize(disease_mask, (leaf_mask.shape[1], leaf_mask.shape[0]))
    
    # Eşik değeri uygula (beyaz pikselleri bul)
    _, leaf_binary = cv2.threshold(leaf_mask, 254, 255, cv2.THRESH_BINARY)
    _, disease_binary = cv2.threshold(disease_mask, 5, 255, cv2.THRESH_BINARY)
    
    # Beyaz piksel sayılarını hesapla
    leaf_pixels = np.sum(leaf_binary == 255)
    disease_pixels = np.sum(disease_binary == 255)
    
    # Hastalık yüzdesini hesapla
    disease_percentage = (disease_pixels / leaf_pixels) * 100 if leaf_pixels > 0 else 0
    
    return leaf_pixels, disease_pixels, disease_percentage

def segmentation_view(request):
    img_url = request.GET.get('image')
    if not img_url:
        return redirect('home')

    # Görüntü adını al
    image_name = os.path.basename(img_url)
    
    # Görüntüyü veritabanından al
    img_path = get_image_from_db(image_name)
    if not img_path:
        return redirect('home')

    # Segmentasyon
    out_dir = os.path.join(settings.MEDIA_ROOT, 'seg_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Yaprak segmentasyonu
    leaf_mask_path = segment_leaf(
        img_path,
        WHOLE_LEAF_MODEL,
        output_path=os.path.join(out_dir, f'leaf_mask_{image_name}')
    )

    # Hastalık segmentasyonu
    disease_mask_path = segment_disease(
        img_path,
        DISEASE_MODEL,
        output_path=os.path.join(out_dir, f'disease_mask_{image_name}')
    )

    # Overlay görüntü oluştur
    overlay_path = create_overlay(
        img_path,
        disease_mask_path,
        output_path=os.path.join(out_dir, f'overlay_{image_name}')
    )

    # Hastalık yüzdesini hesapla
    leaf_pixels, disease_pixels, disease_percentage = calculate_disease_percentage(
        leaf_mask_path, disease_mask_path
    )

    # URL formatına çevir
    def to_url(path):
        return f'/media/seg_outputs/{os.path.basename(path)}'

    return render(request, 'classifier/seg.html', {
        'original_image': f'/media/temp/{image_name}',
        'leaf_mask': to_url(leaf_mask_path),
        'disease_mask': to_url(disease_mask_path),
        'overlay_image': to_url(overlay_path),
        'leaf_pixels': leaf_pixels,
        'disease_pixels': disease_pixels,
        'disease_percentage': disease_percentage,
    })
    