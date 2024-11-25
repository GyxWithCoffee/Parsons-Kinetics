from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt # type: ignore
from .models import Consulta
from .forms import CrearConsultaForm
from .calculos import wind_analysis



def index(request):
    return render(request,'index.html')

def formulario(request):
    return render(request,'formulario.html')

def lista_estimaciones(request):
    consultas = Consulta.objects.all()
    return render(request,'lista_estimaciones.html',{
        'consultas':consultas
    })


def consulta(request):
    if request.method == 'POST':  # Manejar solicitud POST
        form = CrearConsultaForm(request.POST)  # Pasar datos POST al formulario
        if form.is_valid():  # Validar el formulario
            # Guardar el formulario y obtener el objeto creado
            consulta_obj = form.save()

            try:
                # Llamar a la función wind_analysis con los datos del formulario
                resultados = wind_analysis(
                    lat=consulta_obj.lat,
                    lon=consulta_obj.lon,
                    start_date=consulta_obj.start_date,
                    end_date=consulta_obj.end_date
                )

                # Actualizar el objeto Consulta con los resultados
                # Datos
                consulta_obj.elevation = resultados['datos']['Elevation']
                consulta_obj.air_density = resultados['datos']['air_density']

                # Gráficas en formato binario
                consulta_obj.wind_speed_timeseries = resultados['Series Temporales'].getvalue()
                consulta_obj.wind_speed_distribution = resultados['Distribucion Velocidad'].getvalue()
                consulta_obj.wind_distribucion_acumulada = resultados['Distribucion Acumulada'].getvalue()
                consulta_obj.wind_rose_10m = resultados['Roseta 10m'].getvalue()
                consulta_obj.wind_rose_100m = resultados['Roseta 100m'].getvalue()
                consulta_obj.weibull_curves = resultados['Curva Weibull']['Grafica'].getvalue()
                consulta_obj.weibull_distribution = resultados['Distribucion Weibull']['Grafica'].getvalue()
                consulta_obj.eficiencia_generador = resultados['Eficiencia Generador']['Grafica'].getvalue()
                consulta_obj.potencia_viento = resultados['Potencia Viento'].getvalue()
                consulta_obj.potencia_turbina = resultados['Potencia Turbina'].getvalue()
                consulta_obj.potencia_acumulada = resultados['Potencia Acumulada']['Grafica'].getvalue()

                # Parámetros y coeficientes
                consulta_obj.weibull_params_10m = resultados['Curva Weibull']['Parametros 10m']
                consulta_obj.weibull_params_100m = resultados['Curva Weibull']['Parametros 100m']
                consulta_obj.weibull_params_hm = resultados['Distribucion Weibull']['Parametros']
                consulta_obj.eficiencia_coefficients = resultados['Eficiencia Generador']['Coeficientes']

                # Potencia acumulada
                consulta_obj.potencia_max = resultados['Potencia Acumulada']['Max']
                consulta_obj.potencia_promedio = resultados['Potencia Acumulada']['Promedio']

                # Guardar los resultados adicionales en la base de datos
                consulta_obj.save()

                # Redirigir al índice tras éxito
                return redirect('index')

            except Exception as e:
                # Manejar errores de la función wind_analysis
                form.add_error(None, f"Error al procesar el análisis: {str(e)}")
                consulta_obj.delete()  # Eliminar la consulta si falló el análisis

    else:  # Manejar solicitud GET
        form = CrearConsultaForm()  # Crear un formulario vacío

    return render(request, 'formulario.html', {'form': form})


