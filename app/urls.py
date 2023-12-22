from django.urls import path
from . import views

urlpatterns = [
    path('',views.HomeView.as_view(),name='home'),
    path('CSStoTailwind',views.cssToTailwindView.as_view(),name='css_to_tailwind'),
    path('nextAI',views.nextAIView.as_view(),name='nextai'),
    path('generate-stream',views.StreamGeneratorView.as_view(),name='generate_stream'),
]