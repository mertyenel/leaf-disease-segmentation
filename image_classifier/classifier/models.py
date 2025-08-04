# classifier/models.py

from django.db import models

class LeafImage(models.Model):
    image_name = models.CharField(max_length=255)
    image_data = models.BinaryField()
    type = models.CharField(max_length=255)  # Cherry, Peach, Strawberry
    condition = models.CharField(max_length=255)  # Healthy, Unhealthy
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - {self.type} ({self.condition})"

    class Meta:
        db_table = 'leaf_images'