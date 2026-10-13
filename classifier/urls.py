from django.urls import path
from . import views

urlpatterns = [
    # Public & Auth
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # User flows
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('result/<int:pk>/demo/', views.result_demo_view, name='result_demo'),
    path('history/', views.history_view, name='history'),

    # Ontology & Search
    path('ontology/', views.ontology_view, name='ontology'),
    path('search/', views.search_view, name='search'),

    # Evaluation & Performance
    path('performance/', views.performance_view, name='performance'),

    # Admin workflows
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-dashboard/dataset/', views.dataset_manager_view, name='dataset_manager'),
]
