from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name= 'index'),
    path('formulario/',views.index, name= 'formulario'),
    path('Estimacion_creada/',views.consulta, name= 'consulta'),
    path('lista_estimaciones/',views.lista_estimaciones, name= 'lista'),
    path('eliminar-consulta/<int:consulta_id>/', views.eliminar_consulta, name='eliminar_consulta'),

]