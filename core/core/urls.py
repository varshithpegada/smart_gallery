from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from gallery import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')), # Built-in Login/Logout
    path('register/', views.register, name='register'),
    
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('people/', views.people, name='people'),
    path('people/<int:group_id>/', views.person_detail, name='person_detail'),
    path('rescan/', views.rescan_all, name='rescan'),
    path('debug/', views.debug_stats),
    path('photo/<int:photo_id>/', views.photo_detail, name='photo_single'),
    path('photo/<int:photo_id>/delete/', views.delete_photo, name='delete_photo'),
    path('people/<int:group_id>/collage/', views.generate_person_collage, name='create_collage'),
    path('search/', views.image_search, name='image_search'),
    path('events/', views.events_list, name='events'),
    path('events/<int:event_id>/', views.event_detail, name='event_detail'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)