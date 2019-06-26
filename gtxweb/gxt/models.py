from django.db import models

class Person(models.Model):
    picture = CameraField(upload_to='photo', blank=True)
