from datetime import timedelta
from django.utils import timezone
from .models import Photo, Event


def auto_generate_events(user):
    photos = Photo.objects.filter(user=user).order_by('uploaded_at')
    if not photos.exists(): return

    TIME_GAP_THRESHOLD = timedelta(hours=4)
    current_batch = []
    last_time = None

    Event.objects.filter(user=user).delete()

    for photo in photos:
        if last_time is None:
            current_batch.append(photo)
            last_time = photo.uploaded_at
            continue

        gap = photo.uploaded_at - last_time
        if gap < TIME_GAP_THRESHOLD:
            current_batch.append(photo)
            last_time = photo.uploaded_at
        else:
            save_event_batch(user, current_batch)
            current_batch = [photo]
            last_time = photo.uploaded_at

    if current_batch:
        save_event_batch(user, current_batch)


def save_event_batch(user, photos):
    if not photos: return

    first_date = photos[0].uploaded_at
    hour = first_date.hour

    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 22:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    title = f"{first_date.strftime('%b %d')} • {time_of_day}"

    event = Event.objects.create(user=user, title=title, date=first_date.date())

    for p in photos:
        p.event = event
        p.save()