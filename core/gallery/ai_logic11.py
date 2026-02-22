import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image, ImageOps
from .models import Photo, Face, PersonGroup

# Initialize AI Model (Loads once when server starts)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def compute_cosine_distance(embed1, embed2):
    """
    Calculates distance between two faces.
    0.0 = Identical
    1.0 = Completely Different
    """
    dot = np.dot(embed1, embed2)
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    similarity = dot / (norm1 * norm2)
    return 1 - similarity


def scan_photo(photo_instance):
    """
    1. Loads Image & Fixes Rotation
    2. Detects Faces
    3. Saves Face Objects to DB
    4. Returns list of newly created Face objects
    """
    img_path = photo_instance.image.path
    new_faces_found = []

    try:
        # Load with PIL to handle rotation (EXIF)
        pil_image = Image.open(img_path)
        pil_image = ImageOps.exif_transpose(pil_image)

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return []

    # Detect
    faces = face_app.get(img_bgr)

    for idx, face_data in enumerate(faces):
        # Serialize Vector
        encoding_bytes = pickle.dumps(face_data.embedding)

        # Calculate Crop
        bbox = face_data.bbox.astype(int)
        h, w, _ = img_rgb.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])

        face_crop = img_rgb[y1:y2, x1:x2]

        if face_crop.size == 0: continue

        # Create Thumbnail
        buffer = BytesIO()
        Image.fromarray(face_crop).save(buffer, format='JPEG', quality=90)

        # Save to DB
        new_face = Face(
            photo=photo_instance,
            encoding=encoding_bytes,
            location=bbox.tolist()  # Important for Smart Collage
        )
        new_face.image_cutout.save(
            f"face_{photo_instance.id}_{idx}.jpg",
            ContentFile(buffer.getvalue()),
            save=False
        )
        new_face.save()

        new_faces_found.append(new_face)

    return new_faces_found


def assign_face_to_existing_group(face_instance, user):
    """
    FAST METHOD (For Uploads):
    Compares ONE face against existing groups.
    Does NOT reshuffle the whole database.
    """
    new_encoding = pickle.loads(face_instance.encoding)

    # Get all groups for this user
    existing_groups = PersonGroup.objects.filter(user=user)

    best_match_group = None
    best_distance = 1.0
    THRESHOLD = 0.45  # Slightly looser than DBSCAN (0.4) to catch variations

    # Check against every group
    for group in existing_groups:
        # We compare against the first face in that group
        representative_face = group.faces.first()

        if representative_face:
            rep_encoding = pickle.loads(representative_face.encoding)
            dist = compute_cosine_distance(new_encoding, rep_encoding)

            if dist < best_distance:
                best_distance = dist
                best_match_group = group

    # Assign or Create
    if best_distance < THRESHOLD and best_match_group:
        # Match Found
        face_instance.person_group = best_match_group
        face_instance.save()
        # print(f"Assigned to {best_match_group.name} (Dist: {best_distance:.2f})")
    else:
        # No Match - Create New Person
        count = PersonGroup.objects.filter(user=user).count()
        new_group = PersonGroup.objects.create(
            user=user,
            name=f"Person {count + 1}",
            cover_face=face_instance.image_cutout
        )
        face_instance.person_group = new_group
        face_instance.save()
        # print(f"Created New Group: {new_group.name}")


def full_clustering_dbscan(user):
    """
    SLOW METHOD (For Rescan):
    Takes ALL faces and regroups them from scratch using DBSCAN.
    """
    all_faces = Face.objects.filter(photo__user=user)
    count = all_faces.count()
    if count == 0: return

    # 1. Reset all current assignments
    # (Optional: You might want to keep names, but for a true rescan we reset)
    # For this mini-project, let's keep it simple and just re-cluster ids
    encodings = [pickle.loads(f.encoding) for f in all_faces]
    X = np.array(encodings)

    # 2. Run DBSCAN
    # metric='cosine', eps=0.4 is the sweet spot for faces
    clt = DBSCAN(eps=0.4, min_samples=1, metric="cosine")
    clt.fit(X)

    # 3. Apply results
    # First, detach everyone to start clean
    for f in all_faces:
        f.person_group = None
        f.save()

    # Remove empty groups
    PersonGroup.objects.filter(user=user, faces=None).delete()

    unique_labels = np.unique(clt.labels_)

    for label_id in unique_labels:
        if label_id == -1: continue  # Noise

        indices = np.where(clt.labels_ == label_id)[0]
        cluster_faces = [all_faces[int(i)] for i in indices]

        # Create or Get Group
        # We reuse existing logic: find if a group exists for this cluster label
        # (In production you map cluster IDs to Group IDs)

        group_name = f"Person {label_id}"
        group, created = PersonGroup.objects.get_or_create(user=user, name=group_name)

        for f in cluster_faces:
            f.person_group = group
            f.save()

            if not group.cover_face:
                group.cover_face = f.image_cutout
                group.save()


def search_photos_by_face(uploaded_image_file, user):
    """
    Takes an uploaded image, finds the face, and returns
    all existing photos of that person from the DB.
    """
    try:
        # 1. Load the uploaded image into memory (standard preprocessing)
        pil_image = Image.open(uploaded_image_file)
        pil_image = ImageOps.exif_transpose(pil_image)
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 2. Detect face in the QUERY image
        faces = face_app.get(img_bgr)

        if len(faces) == 0:
            return []  # No face found in query

        # We take the largest face in the query image as the target
        query_embedding = faces[0].embedding

        # 3. Compare against Database
        # Optimization: In production, use a Vector DB. For now, we loop.
        all_db_faces = Face.objects.filter(photo__user=user)

        matched_photo_ids = set()
        THRESHOLD = 0.45  # Similarity threshold

        for db_face in all_db_faces:
            # Load stored vector
            db_encoding = pickle.loads(db_face.encoding)

            # Calculate distance
            dist = compute_cosine_distance(query_embedding, db_encoding)

            if dist < THRESHOLD:
                matched_photo_ids.add(db_face.photo.id)

        # 4. Fetch actual Photo objects
        return Photo.objects.filter(id__in=matched_photo_ids).order_by('-uploaded_at')

    except Exception as e:
        print(f"Search Error: {e}")
        return []
