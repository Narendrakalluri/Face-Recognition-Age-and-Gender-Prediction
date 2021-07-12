from django.conf.urls import url, include
from age_and_gender import views


urlpatterns = [
    url(r'age_gender', views.age_gender),
    url(r'gendercam_feed', views.gendercam_feed, name='gendercam_feed'),
]