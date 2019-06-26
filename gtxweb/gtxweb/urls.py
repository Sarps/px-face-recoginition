
from django.contrib import admin
from django.urls import path
from gtx import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.image, name='image')
]
