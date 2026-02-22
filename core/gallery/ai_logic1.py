import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image, ImageOps  # Added ImageOps
from .models import Photo, Face, PersonGroup

# Initialize AI
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def scan_photo(photo_instance):
    """Detects faces - Handles Rotation & Path Issues"""
    img_path = photo_instance.image.path

    try:
        # 1. Open with PIL first (Better at handling paths & rotation than OpenCV)
        pil_image = Image.open(img_path)

        # 2. Fix Rotation (Phone photos are often sideways internally)
        pil_image = ImageOps.exif_transpose(pil_image)

        # 3. Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 4. Convert to Numpy array for AI (RGB format)
        img_rgb = np.array(pil_image)

        # 5. Convert RGB to BGR (InsightFace expects BGR)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return

    # 6. Run Detection
    faces = face_app.get(img_bgr)

    print(f"Photo {photo_instance.id}: Found {len(faces)} faces")  # Debug print

    for idx, face_data in enumerate(faces):
        # Serialize vector
        encoding_bytes = pickle.dumps(face_data.embedding)

        # Crop face using the Corrected Image
        bbox = face_data.bbox.astype(int)
        h, w, _ = img_rgb.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])

        face_crop = img_rgb[y1:y2, x1:x2]  # Crop from RGB image

        if face_crop.size == 0: continue

        # Save thumbnail
        face_pil_out = Image.fromarray(face_crop)
        buffer = BytesIO()
        face_pil_out.save(buffer, format='JPEG', quality=90)

        # Save to DB
        # new_face = Face(photo=photo_instance, encoding=encoding_bytes)
        # NEW CODE: (Pass the location)
        new_face = Face(
            photo=photo_instance,
            encoding=encoding_bytes,
            location=bbox.tolist()  # <--- WE ADDED THIS
        )

        new_face.image_cutout.save(
            f"face_{photo_instance.id}_{idx}.jpg",
            ContentFile(buffer.getvalue()),
            save=False
        )
        new_face.save()


def cluster_faces(user):
    """Groups faces using Cosine Similarity (Better for faces)"""
    # 1. Get all faces
    all_faces = Face.objects.filter(photo__user=user)
    count = all_faces.count()
    if count == 0: return

    # 2. Load encodings
    encodings = [pickle.loads(f.encoding) for f in all_faces]
    X = np.array(encodings)

    # --- CRITICAL FIX HERE ---
    # metric="cosine": Compares the angle/features rather than just distance.
    # eps=0.4: This is the "Distance Threshold".
    #          Lower (0.3) = Very Strict (Must be identical).
    #          Higher (0.5) = Very Loose (Might group lookalikes).
    #          0.4 is the sweet spot for InsightFace.
    clt = DBSCAN(eps=0.4, min_samples=1, metric="cosine")

    clt.fit(X)

    # 3. Save Groups (Same logic as before)
    # Reset existing group assignments first to allow re-shuffling
    for f in all_faces:
        f.person_group = None
        f.save()

    # Delete old empty groups to clean up
    PersonGroup.objects.filter(user=user, faces=None).delete()

    unique_labels = np.unique(clt.labels_)

    for label_id in unique_labels:
        if label_id == -1: continue

        indices = np.where(clt.labels_ == label_id)[0]
        cluster_faces = [all_faces[int(i)] for i in indices]

        # Check if we can merge into an existing group (by name) or create new
        # For simplicity in this logic, we create/get based on the label ID
        # In a real app, you would track "Cluster 0 = Mom" permanently.

        group_name = f"Person {label_id}"

        # Try to find a group that already has these faces, or create new
        group, created = PersonGroup.objects.get_or_create(user=user, name=group_name)

        for f in cluster_faces:
            f.person_group = group
            f.save()
            # Set cover if missing
            if not group.cover_face:
                group.cover_face = f.image_cutout
                group.save()