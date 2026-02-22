from django.shortcuts import render

# Create your views here.
import threading
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, PhotoUploadForm
from .models import Photo, PersonGroup
from .ai_logic import scan_photo, cluster_faces
from django.http import HttpResponse
from django.db.models import Count #
from django.db import transaction # For safety

import random
from io import BytesIO
from django.core.files.base import ContentFile
from .collage_maker import create_collage


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    return render(request, 'registration/register.html', {'form': form})


@login_required
def home(request):
    photos = Photo.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'gallery/home.html', {'photos': photos})


@login_required
def upload(request):
    if request.method == 'POST':
        # Retrieve the list of files from the input named 'image'
        images = request.FILES.getlist('image')

        if images:
            new_photos = []

            # 1. Save all photos to database first
            with transaction.atomic():
                for image_file in images:
                    photo = Photo.objects.create(user=request.user, image=image_file)
                    new_photos.append(photo)

            # 2. Define the background task for the whole batch
            def batch_process(photos, user):
                print(f"Batch processing {len(photos)} photos...")
                # Phase 1: Detect Faces in all new photos
                for p in photos:
                    scan_photo(p)

                # Phase 2: Cluster everything ONCE
                print("Running Clustering...")
                cluster_faces(user)
                print("Batch Finished.")

            # 3. Start Background Thread
            threading.Thread(target=batch_process, args=(new_photos, request.user)).start()

            return redirect('home')

    # GET request: just show the empty form
    form = PhotoUploadForm()
    return render(request, 'gallery/upload.html', {'form': form})


@login_required
def people(request):
    # Old Code:
    # groups = PersonGroup.objects.filter(user=request.user).exclude(faces=None).distinct()

    # New Code: Sort by "Most Photos First"
    groups = PersonGroup.objects.filter(user=request.user) \
        .annotate(num_faces=Count('faces')) \
        .filter(num_faces__gt=0) \
        .order_by('-num_faces')

    return render(request, 'gallery/people.html', {'groups': groups})


@login_required
def person_detail(request, group_id):
    group = get_object_or_404(PersonGroup, id=group_id, user=request.user)

    if request.method == "POST":
        new_name = request.POST.get("name")
        if new_name:
            group.name = new_name
            group.save()

    # Get all photos containing this person
    faces = group.faces.all()
    photos = set([f.photo for f in faces])

    return render(request, 'gallery/person_detail.html', {'group': group, 'photos': photos})


@login_required
def rescan_all(request):
    # 1. Clear existing AI data to start fresh
    from .models import Face, PersonGroup
    Face.objects.filter(photo__user=request.user).delete()
    PersonGroup.objects.filter(user=request.user).delete()

    # 2. Rescan every photo
    photos = Photo.objects.filter(user=request.user)
    for photo in photos:
        scan_photo(photo)

    # 3. Cluster
    cluster_faces(request.user)

    return redirect('people')

# Add this view function
def debug_stats(request):
    f_count = Face.objects.filter(photo__user=request.user).count()
    g_count = PersonGroup.objects.filter(user=request.user).count()
    return HttpResponse(f"<h1>Debug Stats</h1><p>Faces Detected in DB: {f_count}</p><p>Groups Created: {g_count}</p>")


@login_required
def photo_detail(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id, user=request.user)

    # 1. Get all faces in this photo that are assigned to a group
    faces_in_photo = photo.faces.filter(person_group__isnull=False)

    # 2. Get the list of PersonGroup IDs found in this photo
    group_ids = [f.person_group.id for f in faces_in_photo]

    # 3. Find "Similar Photos":
    # Photos that contain ANY of the people found in the current photo
    # .exclude(id=photo.id) ensures we don't show the current photo in the suggestions
    similar_photos = Photo.objects.filter(
        user=request.user,
        faces__person_group__id__in=group_ids
    ).exclude(id=photo.id).distinct().order_by('-uploaded_at')[:12]  # Limit to 12 suggestions

    return render(request, 'gallery/photo_detail.html', {
        'photo': photo,
        'faces': faces_in_photo,
        'similar_photos': similar_photos
    })


@login_required
def delete_photo(request, photo_id):
    # 1. Get the photo (ensure it belongs to the current user)
    photo = get_object_or_404(Photo, id=photo_id, user=request.user)

    if request.method == "POST":
        # 2. Delete it (This cascades and deletes the associated Faces too)
        photo.delete()
        return redirect('home')

    # If accessed via GET, just go back home
    return redirect('home')


@login_required
def generate_person_collage(request, group_id):
    group = get_object_or_404(PersonGroup, id=group_id, user=request.user)

    # 1. Get random photos of this person
    faces = group.faces.all()
    # Get distinct photos (avoid duplicates)
    all_photos = list(set([f.photo for f in faces]))

    if len(all_photos) < 3:
        # Not enough photos
        return redirect('person_detail', group_id=group.id)

    # Pick 3, 4, or 5 random photos
    num_to_pick = min(len(all_photos), 5)
    selected_photos = random.sample(all_photos, num_to_pick)

    # 2. Run AI Collage Maker
    collage_image = create_collage(selected_photos)

    if collage_image:
        # 3. Save as a new Photo in the gallery
        buffer = BytesIO()
        collage_image.save(buffer, format='JPEG', quality=95)

        new_photo = Photo(user=request.user)
        new_photo.image.save(
            f"collage_{group.name}_{random.randint(1000, 9999)}.jpg",
            ContentFile(buffer.getvalue())
        )
        new_photo.save()

        # Optional: Run AI on the collage itself to find faces again!
        # scan_photo(new_photo)

    return redirect('home')