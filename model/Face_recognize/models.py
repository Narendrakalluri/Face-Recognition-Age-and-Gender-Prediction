from __future__ import unicode_literals
from datetime import datetime
from django.db import models

# Create your models here.
class Face_recognize(models.Model):
    userName = models.CharField(max_length=50)

    recorded_at = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.userName
    class Meta:
        verbose_name_plural = "Face_recognize"
