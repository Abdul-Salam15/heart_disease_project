from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('results/<int:prediction_id>/', views.results_view, name='results'),
    path('blockchain/', views.blockchain_view, name='blockchain'),
    path('history/', views.history_view, name='history'),
]
