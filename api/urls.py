from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name= 'index'),
    path('formulario/',views.index, name= 'formulario'),
    path('consulta/',views.consulta, name= 'consulta'),
    path('lista_estimaciones/',views.lista_estimaciones, name= 'lista'),
    
]