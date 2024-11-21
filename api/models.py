from django.db import models

class Consulta(models.Model):
    name = models.CharField(max_length=200)  # Nombre de la ubicación
    lat = models.DecimalField(max_digits=10, decimal_places=7)  # Hasta 7 decimales
    lon = models.DecimalField(max_digits=11, decimal_places=7)  # Hasta 7 decimales
    start_date = models.DateField()  # Fecha de inicio
    end_date = models.DateField()  # Fecha final
    created_at = models.DateTimeField(auto_now_add=True)  # Fecha de creación automática

    # Datos
    elevation = models.FloatField(null=True, blank=True)  # Elevación en metros
    air_density = models.FloatField(null=True, blank=True)  # Densidad del aire

    # Gráficas en formato binario
    wind_speed_timeseries = models.BinaryField(null=True, blank=True)  # Serie temporal
    wind_speed_distribution = models.BinaryField(null=True, blank=True)  # Distribución de velocidad
    wind_distribucion_acumulada = models.BinaryField(null=True, blank=True)  # Distribución acumulada
    wind_rose_10m = models.BinaryField(null=True, blank=True)  # Roseta a 10m
    wind_rose_100m = models.BinaryField(null=True, blank=True)  # Roseta a 100m
    weibull_curves = models.BinaryField(null=True, blank=True)  # Curvas de Weibull
    weibull_distribution = models.BinaryField(null=True, blank=True)  # Distribución Weibull
    eficiencia_generador = models.BinaryField(null=True, blank=True)  # Eficiencia generador
    potencia_viento = models.BinaryField(null=True, blank=True)  # Potencia viento
    potencia_turbina = models.BinaryField(null=True, blank=True)  # Potencia turbina
    potencia_acumulada = models.BinaryField(null=True, blank=True)  # Potencia acumulada

    # Parámetros y coeficientes
    weibull_params_10m = models.JSONField(null=True, blank=True)  # Parámetros de Weibull a 10m
    weibull_params_100m = models.JSONField(null=True, blank=True)  # Parámetros de Weibull a 100m
    weibull_params_hm = models.JSONField(null=True, blank=True)  # Parámetros de Weibull interpolados a h
    eficiencia_coefficients = models.JSONField(null=True, blank=True)  # Coeficientes de eficiencia

    # Potencia acumulada
    potencia_max = models.FloatField(null=True, blank=True)  # Potencia máxima
    potencia_promedio = models.FloatField(null=True, blank=True)  # Potencia promedio

    def __str__(self):
        return f"{self.name} ({self.lat}, {self.lon})"
