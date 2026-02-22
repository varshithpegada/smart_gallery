import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image, ImageOps
from .models import Photo, Face, PersonGroup

# Initialize AI Model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def compute_cosine_distance(embed1, embed2):
    dot = np.dot(embed1, embed2)
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    similarity = dot / (norm1 * norm2)
    return 1 - similarity


def scan_photo(photo_instance):
    """Detects faces, saves them to DB, and returns list of new Face objects."""
    img_path = photo_instance.image.path
    new_faces_found = []

    try:
        pil_image = Image.open(img_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return []

    faces = face_app.get(img_bgr)

    for idx, face_data in enumerate(faces):
        encoding_bytes = pickle.dumps(face_data.embedding)
        bbox = face_data.bbox.astype(int)
        h, w, _ = img_rgb.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
        face_crop = img_rgb[y1:y2, x1:x2]

        if face_crop.size == 0: continue

        buffer = BytesIO()
        Image.fromarray(face_crop).save(buffer, format='JPEG', quality=90)

        new_face = Face(
            photo=photo_instance,
            encoding=encoding_bytes,
            location=bbox.tolist()
        )
        new_face.image_cutout.save(f"face_{photo_instance.id}_{idx}.jpg", ContentFile(buffer.getvalue()), save=False)
        new_face.save()
        new_faces_found.append(new_face)

    return new_faces_found


def assign_face_to_existing_group(face_instance, user):
    """Fast Incremental Logic for Uploads"""
    new_encoding = pickle.loads(face_instance.encoding)
    existing_groups = PersonGroup.objects.filter(user=user)
    best_match_group = None
    best_distance = 1.0
    THRESHOLD = 0.45

    for group in existing_groups:
        representative_face = group.faces.first()
        if representative_face:
            rep_encoding = pickle.loads(representative_face.encoding)
            dist = compute_cosine_distance(new_encoding, rep_encoding)
            if dist < best_distance:
                best_distance = dist
                best_match_group = group

    if best_distance < THRESHOLD and best_match_group:
        face_instance.person_group = best_match_group
        face_instance.save()
    else:
        count = PersonGroup.objects.filter(user=user).count()
        new_group = PersonGroup.objects.create(
            user=user,
            name=f"Person {count + 1}",
            cover_face=face_instance.image_cutout
        )
        face_instance.person_group = new_group
        face_instance.save()


def full_clustering_dbscan(user):
    """Slow Logic for Rescan"""
    all_faces = Face.objects.filter(photo__user=user)
    if all_faces.count() == 0: return

    encodings = [pickle.loads(f.encoding) for f in all_faces]
    X = np.array(encodings)

    clt = DBSCAN(eps=0.4, min_samples=1, metric="cosine")
    clt.fit(X)

    for f in all_faces:
        f.person_group = None
        f.save()

    PersonGroup.objects.filter(user=user, faces=None).delete()

    unique_labels = np.unique(clt.labels_)
    for label_id in unique_labels:
        if label_id == -1: continue
        indices = np.where(clt.labels_ == label_id)[0]
        cluster_faces = [all_faces[int(i)] for i in indices]

        group, created = PersonGroup.objects.get_or_create(user=user, name=f"Person {label_id}")
        for f in cluster_faces:
            f.person_group = group
            f.save()
            if not group.cover_face:
                group.cover_face = f.image_cutout
                group.save()


def search_photos_by_face(uploaded_image_file, user):
    """Google Lens Style Search Logic"""
    try:
        pil_image = Image.open(uploaded_image_file)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        faces = face_app.get(img_bgr)
        if len(faces) == 0: return []

        query_embedding = faces[0].embedding
        all_db_faces = Face.objects.filter(photo__user=user)
        matched_photo_ids = set()

        for db_face in all_db_faces:
            db_encoding = pickle.loads(db_face.encoding)
            dist = compute_cosine_distance(query_embedding, db_encoding)
            if dist < 0.45:
                matched_photo_ids.add(db_face.photo.id)

        return Photo.objects.filter(id__in=matched_photo_ids).order_by('-uploaded_at')
    except Exception as e:
        print(f"Search Error: {e}")
        return []