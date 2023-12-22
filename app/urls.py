from django.urls import path
from . import views

urlpatterns = [
    path('',views.HomeView.as_view(),name='home'),
    path('generate-stream',views.StreamGeneratorView.as_view(),name='generate_stream'),
]