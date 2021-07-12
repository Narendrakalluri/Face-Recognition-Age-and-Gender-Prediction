from django.conf.urls import url, include
from Face_recognize import views as app_views


urlpatterns = [
    url(r'^$', app_views.index),
    url(r'recognition_home', app_views.recognition_home),
    url(r'recognition', app_views.recognition),
    url(r'home', app_views.home),
    url(r'facecam_feed', app_views.facecam_feed, name='facecam_feed'),
    url(r'^create_dataset$', app_views.create_dataset),    
]
    