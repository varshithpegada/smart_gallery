import random
from PIL import Image, ImageOps
from .models import Face


def smart_crop(photo_instance, target_width, target_height):
    """
    Crops an image while keeping the face in the center.
    """
    img_path = photo_instance.image.path
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # Fix rotation

    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_w, img_h = img.size

    # 1. Find the Face Center (Centroid)
    # We get the first face detected in this photo
    face = Face.objects.filter(photo=photo_instance).first()

    if face and face.location:
        # location is saved as [x1, y1, x2, y2]
        x1, y1, x2, y2 = face.location
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
    else:
        # Fallback to image center if no face data found
        face_center_x = img_w / 2
        face_center_y = img_h / 2

    # 2. Calculate Crop Box
    target_ratio = target_width / target_height
    current_ratio = img_w / img_h

    if current_ratio > target_ratio:
        # Image is wider than target
        new_w = img_h * target_ratio
        new_h = img_h

        # Center horizontally around face
        left = max(0, min(img_w - new_w, face_center_x - (new_w / 2)))
        top = 0
    else:
        # Image is taller than target
        new_w = img_w
        new_h = img_w / target_ratio

        # Center vertically around face
        left = 0
        top = max(0, min(img_h - new_h, face_center_y - (new_h / 2)))

    right = left + new_w
    bottom = top + new_h

    # 3. Perform Crop & Resize
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return img


def create_collage(photos):
    """
    Generates a collage based on number of input photos (3 to 5)
    """
    if len(photos) < 3: return None  # Need at least 3

    # Canvas Settings
    canvas_size = 1200  # 1200x1200px output
    gap = 20  # White space between images
    canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')

    count = len(photos)

    # --- LAYOUT LOGIC ---

    if count == 3:
        # Layout: 1 Big Left, 2 Small Right
        # Big Image
        w, h = (canvas_size // 2) - gap, canvas_size - (2 * gap)
        img1 = smart_crop(photos[0], w, h)
        canvas.paste(img1, (gap, gap))

        # Small Images
        w_small, h_small = (canvas_size // 2) - gap, (canvas_size // 2) - (1.5 * gap)
        img2 = smart_crop(photos[1], w_small, int(h_small))
        img3 = smart_crop(photos[2], w_small, int(h_small))

        canvas.paste(img2, (int(canvas_size / 2) + gap, gap))
        canvas.paste(img3, (int(canvas_size / 2) + gap, int(canvas_size / 2) + int(gap / 2)))

    elif count == 4:
        # Layout: 2x2 Grid
        w, h = (canvas_size // 2) - (1.5 * gap), (canvas_size // 2) - (1.5 * gap)
        coords = [
            (gap, gap),
            (int(canvas_size / 2) + int(gap / 2), gap),
            (gap, int(canvas_size / 2) + int(gap / 2)),
            (int(canvas_size / 2) + int(gap / 2), int(canvas_size / 2) + int(gap / 2))
        ]
        for i, photo in enumerate(photos):
            img = smart_crop(photo, int(w), int(h))
            canvas.paste(img, coords[i])

    elif count >= 5:
        # Layout: 1 Big Top, 4 Small Bottom
        # Top Big
        w_big, h_big = canvas_size - (2 * gap), (canvas_size * 0.6) - gap
        img1 = smart_crop(photos[0], int(w_big), int(h_big))
        canvas.paste(img1, (gap, gap))

        # Bottom Strip
        w_small = (canvas_size - (5 * gap)) / 4
        h_small = (canvas_size * 0.4) - (2 * gap)
        y_pos = int(canvas_size * 0.6) + gap

        for i in range(4):
            img = smart_crop(photos[i + 1], int(w_small), int(h_small))
            x_pos = int(gap + (i * (w_small + gap)))
            canvas.paste(img, (x_pos, y_pos))

    return canvas