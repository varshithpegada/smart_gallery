import uuid
import os
from django.db import models
from django.contrib.auth.models import User


# --- Helper Function for Security ---
def get_file_path(instance, filename):
    """
    Generates a random UUID filename.
    Prevents users from guessing image URLs.
    Example result: photos/550e8400-e29b-41d4-a716-446655440000.jpg
    """
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"

    # Organize into specific folders based on the Model
    if isinstance(instance, Photo):
        return os.path.join('photos/', filename)
    elif isinstance(instance, Face):
        return os.path.join('faces/', filename)
    elif isinstance(instance, PersonGroup):
        return os.path.join('covers/', filename)
    else:
        return os.path.join('uploads/', filename)


# --- MODELS ---

class Event(models.Model):
    """
    Represents a group of photos taken around the same time.
    e.g., 'Wedding Day', 'Trip to Goa', 'Sunday Morning'
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)  # e.g. "Dec 8 - Afternoon"
    date = models.DateField()

    def __str__(self):
        return self.title


class PersonGroup(models.Model):
    """Represents a unique person identified by AI"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50, default="Unknown Person")

    # Cover image for the person's album
    cover_face = models.ImageField(upload_to=get_file_path, null=True, blank=True)

    def __str__(self):
        return self.name


class Photo(models.Model):
    """The original uploaded image"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    # Link to an Event (Optional, automatically assigned later)
    event = models.ForeignKey(Event, on_delete=models.SET_NULL, null=True, blank=True, related_name='photos')

    # Secure upload path
    image = models.ImageField(upload_to=get_file_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Photo {self.id} by {self.user.username}"


class Face(models.Model):
    """A single face detected in a photo"""
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='faces')
    person_group = models.ForeignKey(PersonGroup, on_delete=models.SET_NULL, null=True, blank=True,
                                     related_name='faces')

    # Store the 512-dimension AI vector as raw bytes
    encoding = models.BinaryField()

    # Store a small crop of the face for thumbnails
    image_cutout = models.ImageField(upload_to=get_file_path)

    # Stores [x1, y1, x2, y2] coordinates
    # Critical for the 'Smart Collage' feature to crop correctly
    location = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Face in Photo {self.photo.id}"