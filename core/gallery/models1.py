from django.db import models
from django.contrib.auth.models import User


class PersonGroup(models.Model):
    """Represents a unique person found by AI"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50, default="Unknown Person")
    cover_face = models.ImageField(upload_to='covers/', null=True, blank=True)

    def __str__(self):
        return self.name


class Photo(models.Model):
    """The original uploaded image"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='photos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Photo {self.id}"


class Face(models.Model):
    """A single face detected in a photo"""
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='faces')
    person_group = models.ForeignKey(PersonGroup, on_delete=models.SET_NULL, null=True, blank=True,
                                     related_name='faces')

    # Store the 512-dimension AI vector as raw bytes
    encoding = models.BinaryField()
    # Store a small crop of the face for thumbnails
    image_cutout = models.ImageField(upload_to='faces/')

    # --- ADD THIS LINE ---
    # Stores [x1, y1, x2, y2] coordinates
    location = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Face in Photo {self.photo.id}"models.py