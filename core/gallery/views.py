import threading
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Count
from django.core.files.base import ContentFile
from io import BytesIO

from .forms import RegisterForm, PhotoUploadForm
from .models import Photo, PersonGroup, Face, Event  # <--- Added Event here
from .ai_logic import scan_photo, assign_face_to_existing_group, full_clustering_dbscan, search_photos_by_face
from .collage_maker import create_collage
from .event_logic import auto_generate_events  # <--- Correct Import
from django.http import HttpResponse


# --- AUTH ---
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


def home(request):
    # 1. IF NOT LOGGED IN -> SHOW LANDING PAGE
    if not request.user.is_authenticated:
        return render(request, 'landing.html')

    # 2. IF LOGGED IN -> SHOW GALLERY (Your existing code)
    photos = Photo.objects.filter(user=request.user).order_by('-uploaded_at')

    total_photos = photos.count()
    total_people = PersonGroup.objects.filter(user=request.user).count()
    total_events = Event.objects.filter(user=request.user).count()

    context = {
        'photos': photos,
        'stats': {
            'photos': total_photos,
            'people': total_people,
            'events': total_events
        }
    }
    return render(request, 'gallery/home.html', context)


# --- UPLOAD ---
@login_required
def upload(request):
    if request.method == 'POST':
        images = request.FILES.getlist('image')
        if images:
            new_photos = []
            with transaction.atomic():
                for image_file in images:
                    photo = Photo.objects.create(user=request.user, image=image_file)
                    new_photos.append(photo)

            def batch_process(photos, user):
                print(f"Processing {len(photos)} new photos...")
                for p in photos:
                    new_faces = scan_photo(p)
                    for face in new_faces:
                        assign_face_to_existing_group(face, user)

                # OPTIONAL: Regenerate events after upload
                auto_generate_events(user)
                print("Processing complete.")

            threading.Thread(target=batch_process, args=(new_photos, request.user)).start()
            return redirect('home')

    form = PhotoUploadForm()
    return render(request, 'gallery/upload.html', {'form': form})


# --- PEOPLE ---
@login_required
def people(request):
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

        if new_name and new_name != group.name:
            # 1. Check if a group with this name ALREADY exists
            existing_group = PersonGroup.objects.filter(user=request.user, name=new_name).first()

            if existing_group and existing_group != group:
                # --- MERGE LOGIC ---
                print(f"Merging {group.name} into {existing_group.name}...")

                # Move all faces from the current group to the existing target group
                for face in group.faces.all():
                    face.person_group = existing_group
                    face.save()

                # Delete the now empty group (old one)
                group.delete()

                # Redirect user to the main 'Ajay' group
                return redirect('person_detail', group_id=existing_group.id)

            else:
                # --- SIMPLE RENAME LOGIC ---
                group.name = new_name
                group.save()

    faces = group.faces.all()
    photos = list(set([f.photo for f in faces]))

    return render(request, 'gallery/person_detail.html', {
        'group': group,
        'photos': photos
    })


# --- EVENTS (NEW) ---
@login_required
def events_list(request):
    # Trigger grouping logic on page load to ensure it's up to date
    auto_generate_events(request.user)
    events = Event.objects.filter(user=request.user).order_by('-date')
    return render(request, 'gallery/events.html', {'events': events})


@login_required
def event_detail(request, event_id):
    event = get_object_or_404(Event, id=event_id, user=request.user)
    photos = event.photos.all()
    if request.method == "POST":
        new_title = request.POST.get('title')
        if new_title:
            event.title = new_title
            event.save()
    return render(request, 'gallery/event_detail.html', {'event': event, 'photos': photos})


# --- UTILS ---
@login_required
def photo_detail(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id, user=request.user)
    faces_in_photo = photo.faces.filter(person_group__isnull=False)
    group_ids = [f.person_group.id for f in faces_in_photo]
    similar_photos = Photo.objects.filter(user=request.user, faces__person_group__id__in=group_ids).exclude(
        id=photo.id).distinct().order_by('-uploaded_at')[:12]
    return render(request, 'gallery/photo_detail.html',
                  {'photo': photo, 'faces': faces_in_photo, 'similar_photos': similar_photos})


@login_required
def delete_photo(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id, user=request.user)
    if request.method == "POST":
        photo.delete()
        return redirect('home')
    return redirect('home')


@login_required
def rescan_all(request):
    def run_full_rescan(user):
        full_clustering_dbscan(user)
        auto_generate_events(user)  # Also regenerate events

    threading.Thread(target=run_full_rescan, args=(request.user,)).start()
    return redirect('people')


@login_required
def generate_person_collage(request, group_id):
    group = get_object_or_404(PersonGroup, id=group_id, user=request.user)
    faces = group.faces.all()
    all_photos = list(set([f.photo for f in faces]))
    if len(all_photos) < 3: return redirect('person_detail', group_id=group.id)
    num_to_pick = min(len(all_photos), 5)
    selected_photos = random.sample(all_photos, num_to_pick)
    collage_image = create_collage(selected_photos)
    if collage_image:
        buffer = BytesIO()
        collage_image.save(buffer, format='JPEG', quality=95)
        new_photo = Photo(user=request.user)
        new_photo.image.save(f"collage_{group.name}_{random.randint(1000, 9999)}.jpg", ContentFile(buffer.getvalue()))
        new_photo.save()
        scan_photo(new_photo)
    return redirect('home')

def debug_stats(request):
    f_count = Face.objects.filter(photo__user=request.user).count()
    g_count = PersonGroup.objects.filter(user=request.user).count()
    return HttpResponse(f"<h1>Debug Stats</h1><p>Faces Detected in DB: {f_count}</p><p>Groups Created: {g_count}</p>")


@login_required
def image_search(request):
    results = None
    query_image_url = None
    if request.method == "POST" and request.FILES.get('query_img'):
        image_file = request.FILES['query_img']
        results = search_photos_by_face(image_file, request.user)
        import base64
        image_file.seek(0)
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        query_image_url = f"data:image/jpeg;base64,{encoded_string}"
    return render(request, 'gallery/search.html', {'results': results, 'query_image': query_image_url})