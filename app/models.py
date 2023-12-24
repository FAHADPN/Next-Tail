from django.db import models

# Create your models here.
class image(models.Model):
    image = models.ImageField(upload_to='images/')
    prompt = models.CharField(max_length=200)
    def __str__(self):
        return self.image.url