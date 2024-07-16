"""
URL configuration for myapp.
"""

from django.urls import path
from .views import QueryView, index

urlpatterns = [
    path('', index, name='index'),
    path('query/', QueryView.as_view(), name='query'),
]
