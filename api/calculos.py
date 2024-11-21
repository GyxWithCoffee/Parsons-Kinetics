import pandas as pd
import io
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import seaborn as sns
from scipy.stats import weibull_min
import requests_cache
from windrose import WindroseAxes
from openmeteo_requests import Client  # Import corregido
import matplotlib.pyplot as plt
import pandas as pd
import requests_cache
from openmeteo_requests import Client
# from scipy.integrate import cumtrapz


def wind_analysis(lat, lon, start_date, end_date):
    # Función para configurar la sesión con caché y reintentos
    def retry(session, retries=5, backoff_factor=0.2):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # Example usage
    # Crear sesiones con caché y reintentos
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = Client(session=retry_session)

    # Parámetros para la API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m"]
    }

    # Obtener respuesta de la API
    responses = openmeteo.weather_api(url, params=params)

    # Procesar la primera ubicación
    response = responses[0]

    # En vez de guardar un archivo json para los diccionarios guarda los datos en variables
    response_dict = {
            "Latitude": response.Latitude(),
            "Longitude": response.Longitude(),
            "Elevation": response.Elevation(),
            "Timezone": response.Timezone(),
            "TimezoneAbbreviation": response.TimezoneAbbreviation(),
            "UtcOffsetSeconds": response.UtcOffsetSeconds(),
        }
    # print(response_dict)

    # print('\n'+'-'*100+'\n')

    # print(f'Place: {fname}')
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Procesar datos horarios
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_100m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(2).ValuesAsNumpy(),
        "wind_direction_100m": hourly.Variables(3).ValuesAsNumpy()
    }

    # Convertir a DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    # print(hourly_dataframe)
    # Ahora `hourly_dataframe` contiene todos los datos en un formato utilizable para cálculos y gráficos


    def calculate_air_density(altitude):
        # Constants
        rho_0 = 1.225  # sea level standard air density in kg/m^3
        L = 0.0065  # standard temperature lapse rate in K/m
        T_0 = 288.15  # sea level standard temperature in K
        g = 9.80665  # acceleration due to gravity in m/s^2
        R = 287.058  # ideal gas constant for air in J/(kg·K)

        # Air density calculation
        rho = rho_0 * (1 - (L * altitude / T_0)) ** ((g / (R * L)) - 1)
        return rho

    # Example: Calculate the air density at an altitude of 2500 meters
    altitude = response_dict['Elevation']  # in meters
    air_density = calculate_air_density(altitude)

    # print('\n'+'-'*100+'\n')

    # print(f"Air density at {altitude} meters: {air_density} kg/m^3")

    # Generamos las graficas y las guardamos en variables en formato de bites--------------------------------------------------------------
    # Series Temporales de Velocidad del Viento (10m y 100m) ---------------------------------------------------------
    # Configurar el tamaño de la figura
    plt.figure(figsize=(14, 7))
    # Graficar las series temporales
    plt.plot(hourly_dataframe['date'], hourly_dataframe['wind_speed_10m'], label='Velocidad del viento a 10m')
    #plt.plot(hourly_dataframe['date'], hourly_dataframe['wind_speed_100m'], label='Velocidad del viento a 100m')
    # Añadir títulos y etiquetas
    plt.title('Series Temporales de Velocidad del Viento')
    plt.xlabel('Fecha')
    plt.ylabel('Velocidad del Viento (m/s)')
    plt.legend()
    # Guardar la gráfica en memoria como PNG
    wind_speed_timeseries_bytes = io.BytesIO()
    plt.savefig(wind_speed_timeseries_bytes, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    wind_speed_timeseries_bytes.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'wind_speed_timeseries_bytes' y tiene {len(wind_speed_timeseries_bytes.getvalue())} bytes.")


    # Distribuciones de Velocidad del Viento ------------------------------------------------------------------------
    # Configurar el tamaño de la figura
    plt.figure(figsize=(14, 7))
    # Graficar las distribuciones usando seaborn
    sns.histplot(hourly_dataframe['wind_speed_10m'], kde=True, color='blue', label='Velocidad del viento a 10m', bins=30)
    sns.histplot(hourly_dataframe['wind_speed_100m'], kde=True, color='red', label='Velocidad del viento a 100m', bins=30)
    # Añadir títulos y etiquetas
    plt.title('Distribuciones de Velocidad del Viento')
    plt.xlabel('Velocidad del Viento (m/s)')
    plt.ylabel('Frecuencia')
    plt.legend()
    # Guardar la gráfica en memoria como PNG
    wind_speed_distribution = io.BytesIO()
    plt.savefig(wind_speed_distribution, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    wind_speed_distribution.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'wind_speed_distribution' y tiene {len(wind_speed_distribution.getvalue())} bytes.")


    # Distribución Acumulada de Velocidad del Viento -----------------------------------------------------------------
    # Configurar el tamaño de la figura
    plt.figure(figsize=(8, 4))
    # Graficar las distribuciones acumuladas usando seaborn
    sns.ecdfplot(hourly_dataframe['wind_speed_10m'], color='blue', label='Velocidad del viento a 10m')
    sns.ecdfplot(hourly_dataframe['wind_speed_100m'], color='red', label='Velocidad del viento a 100m')
    # Añadir títulos y etiquetas
    plt.title('Distribuciones Acumuladas de Velocidad del Viento')
    plt.xlabel('Velocidad del Viento (m/s)',fontsize=12)
    plt.ylabel('Probabilidad Acumulada',fontsize=12)
    plt.xlim(0,40 )
    plt.legend()
    # Guardar la gráfica en memoria como PNG
    wind_distribucion_acomulada = io.BytesIO()
    plt.savefig(wind_distribucion_acomulada, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    wind_distribucion_acomulada.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'wind_distribucion_acomulada' y tiene {len(wind_distribucion_acomulada.getvalue())} bytes.")

    # Rosetas de Viento para las Direcciones del Viento -------------------------------------------------------
    # Crear la roseta de viento para la dirección del viento a 10m
    plt.figure(figsize=(1, 1))
    ax = WindroseAxes.from_ax()
    ax.bar(hourly_dataframe['wind_direction_10m'], hourly_dataframe['wind_speed_10m'], normed=True, opening=0.8, edgecolor='white')
    ax.set_title('Roseta de Viento a 10m',fontsize=25)
    ax.set_legend()
    # Guardar la gráfica en memoria como PNG
    wind_rose_10m = io.BytesIO()
    plt.savefig(wind_rose_10m, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    wind_rose_10m.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'wind_rose_10m' y tiene {len(wind_rose_10m.getvalue())} bytes.")

    # Crear la roseta de viento para la dirección del viento a 100m
    plt.figure(figsize=(4, 4))
    ax = WindroseAxes.from_ax()
    ax.bar(hourly_dataframe['wind_direction_100m'], hourly_dataframe['wind_speed_100m'], normed=True, opening=0.8, edgecolor='white')
    ax.set_title('Roseta de Viento a 100m' ,fontsize=12)
    ax.set_legend()
    # Guardar la gráfica en memoria como PNG
    wind_rose_100m = io.BytesIO()
    plt.savefig(wind_rose_100m, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    wind_rose_100m.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'wind_rose_100m' y tiene {len(wind_rose_100m.getvalue())} bytes.")

    # Cálculo de la curva de Weibull------------------------------------------
    # Ajustar la distribución de Weibull a los datos de velocidad del viento
    params_10m = weibull_min.fit(hourly_dataframe['wind_speed_10m'], floc=0)
    params_100m = weibull_min.fit(hourly_dataframe['wind_speed_100m'], floc=0)
    # Obtener los parámetros de Weibull
    shape_10m, loc_10m, scale_10m = params_10m
    shape_100m, loc_100m, scale_100m = params_100m
    # Generar una gama de velocidades de viento para la curva de Weibull
    wind_speeds = np.linspace(0, max(hourly_dataframe['wind_speed_10m'].max(), hourly_dataframe['wind_speed_100m'].max()), 100)
    # Calcular la densidad de probabilidad de Weibull para las velocidades de viento
    weibull_pdf_10m = weibull_min.pdf(wind_speeds, shape_10m, loc_10m, scale_10m)
    weibull_pdf_100m = weibull_min.pdf(wind_speeds, shape_100m, loc_100m, scale_100m)
    # Graficar la curva de Weibull
    plt.figure(figsize=(10, 4))
    plt.plot(wind_speeds, weibull_pdf_10m, label='Weibull (10m)', color='blue')
    plt.plot(wind_speeds, weibull_pdf_100m, label='Weibull (100m)', color='red')
    plt.hist(hourly_dataframe['wind_speed_10m'], bins=30, density=True, alpha=0.5, color='blue', label='Histograma (10m)')
    plt.hist(hourly_dataframe['wind_speed_100m'], bins=30, density=True, alpha=0.5, color='red', label='Histograma (100m)')
    plt.title('Distribución de Weibull de la Velocidad del Viento')
    plt.xlabel('Velocidad del Viento (m/s)',fontsize=12)
    plt.ylabel('Densidad de Probabilidad',fontsize=12)
    plt.ylim(0, 0.2)
    plt.xlim(0,40 )
    plt.legend()
    # Mostrar la gráfica
    # print(f"Parámetros de Weibull a 10m: {params_10m[0]} , {params_10m[1]} , {params_10m[2]} ")
    # print(f"Parámetros de Weibull a 100m: {params_100m[0]} , {params_100m[1]} , {params_100m[2]} ")
    # Guardar la gráfica en memoria como PNG
    Weibull_curves = io.BytesIO()
    plt.savefig(Weibull_curves, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    Weibull_curves.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'Weibull_curves' y tiene {len(Weibull_curves.getvalue())} bytes.")

    # Cálculo de la potencia mecánica estimada producida por una turbina ------------------------------------------
    # Datos del radio del aerogenerador y velocidad angular
    R = 10  # Radio en metros
    h = 20  # Altura de la torre en metros
    # Interpolar los parámetros de Weibull a h m de altura
    # Exponente de la ley de potencia
    alpha = 0.143
    shape_hm = (shape_10m + shape_100m) / 2
    scale_hm = ((scale_10m * (h / 10)**alpha) + (scale_100m * (h / 100)**alpha)) / 2
    # Generar una gama de velocidades de viento para la curva de Weibull
    wind_speeds = np.linspace(0, max(hourly_dataframe['wind_speed_10m'].max(), hourly_dataframe['wind_speed_100m'].max()), 100)
    # Calcular la densidad de probabilidad de Weibull para las velocidades de viento a hm
    weibull_pdf_hm = weibull_min.pdf(wind_speeds, shape_hm, loc=0, scale=scale_hm)
    # Gráfica de la curva de Weibull a hm
    plt.figure(figsize=(14, 7))
    plt.plot(wind_speeds, weibull_pdf_hm, label=f'Weibull ({h} m)', color='green')
    plt.hist(hourly_dataframe['wind_speed_10m'], bins=30, density=True, alpha=0.5, color='blue', label='Histograma (10m)')
    plt.hist(hourly_dataframe['wind_speed_100m'], bins=30, density=True, alpha=0.5, color='red', label='Histograma (100m)')
    plt.title(f'Distribución de Weibull de la Velocidad del Viento a {h} m')
    plt.xlabel('Velocidad del Viento (m/s)')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    # Mostrar la gráfica
    parametrosWeibull = (shape_hm, scale_hm)
    # print(f"Parámetros de Weibull a {h} m:", (shape_hm, scale_hm))
    # Guardar la gráfica en memoria como PNG
    weibull_distribution = io.BytesIO()
    plt.savefig(weibull_distribution, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    weibull_distribution.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'weibull_distribution' y tiene {len(weibull_distribution.getvalue())} bytes.")

    # Ajuste de curva de eficiencia del Aerogenerador ---------------------------------------------------
    # Datos de ejemplo extraídos de la gráfica
    # Lambda (λ) es el ratio de velocidad de la punta
    # Cp es el coeficiente de potencia
    lambda_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    cp_values = np.array([0.0, 0.15, 0.3, 0.45, 0.52, 0.55, 0.53, 0.5, 0.4, 0.3, 0.1])
    # Ajustar un polinomio de grado 4 a los datos
    p = Polynomial.fit(lambda_values, cp_values, 4)
    # Obtener los coeficientes del polinomio
    coefficients = p.convert().coef
    # print("Coeficientes del polinomio:", coefficients)
    # Generar datos para la curva ajustada
    lambda_fit = np.linspace(0, 2, 100)
    cp_fit = p(lambda_fit)
    # Graficar los datos originales y la curva ajustada
    plt.scatter(lambda_values, cp_values, label='Datos Originales')
    plt.plot(lambda_fit, cp_fit, label='Curva Ajustada', color='red')
    plt.xlabel('λ')
    plt.ylabel('Cp')
    plt.title('Ajuste de Curva de Eficiencia del Aerogenerador')
    plt.legend()
    # Guardar la gráfica en memoria como PNG
    eficiencia_aerogenerador = io.BytesIO()
    plt.savefig(eficiencia_aerogenerador, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    eficiencia_aerogenerador.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'eficiencia_aerogenerador' y tiene {len(eficiencia_aerogenerador.getvalue())} bytes.")


    # Potencia vs Velocidad del Viento ---------------------------------------------------------------
    # Rango de velocidades del viento
    v_wind = np.linspace(1, 25, 50)  # Velocidad del viento en m/s
    # Definir un rango de valores de Omega creciente (convertir de rpm a rad/s)
    Omega_min = 0 # 0 rpm en rad/s
    Omega_max = 40 * 2 * np.pi / 60  # 180 rpm en rad/s
    Omega_values = np.linspace(Omega_min, Omega_max, len(v_wind))
    # Calcular lambda para cada velocidad del viento
    lambda_values =  (Omega_values * R) /v_wind
    # Calcular Cp para cada velocidad del viento usando el polinomio de Cp(lambda)
    cp_values = p(lambda_values)
    # Graficar Cp vs Velocidad del Viento
    plt.figure(figsize=(10, 6))
    plt.plot(v_wind, cp_values, label='Cp vs Velocidad del Viento')
    plt.xlabel('Velocidad del Viento (m/s)')
    plt.ylabel('Cp')
    plt.title('Coeficiente de Potencia (Cp) vs Velocidad del Viento')
    plt.legend()
    plt.grid(True)
    # Guardar la gráfica en memoria como PNG
    potencia_viento = io.BytesIO()
    plt.savefig(potencia_viento, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    potencia_viento.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'potencia_viento' y tiene {len(potencia_viento.getvalue())} bytes.")


    # Función para calcular la potencia eólica disponible --------------------------------------------
    def power_available(v, air_density, R):
        return 0.5 * air_density * np.pi * R**2 * v**3
    # Función para calcular la potencia generada por la turbina
    def power_generated(v, air_density, R, Omega, p):
        lambda_value = Omega * R / v
        cp = p(lambda_value)
        return power_available(v, air_density, R) * cp
    # Calcular la densidad de potencia eólica para cada velocidad del viento
    Omega = 400 * 2 * np.pi / 60
    power_densities = [power_generated(v, air_density, R, Omega , p) for v in wind_speeds]
    ## print(Omega_values)
    # Graficar la densidad de potencia eólica vs velocidad del viento
    plt.figure(figsize=(10, 6))
    # # print(power_densities*weibull_pdf_hm)
    plt.plot(wind_speeds,power_densities*weibull_pdf_hm, label='Potencia Eólica')
    plt.xlabel('Velocidad del Viento (m/s)')
    plt.ylabel('Potencia (W)')
    plt.title('Potencia Eólica vs Velocidad del Viento')
    plt.legend()
    plt.grid(True)
    # Guardar la gráfica en memoria como PNG
    windturbine_power = io.BytesIO()
    plt.savefig(windturbine_power, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    windturbine_power.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'windturbine_power' y tiene {len(windturbine_power.getvalue())} bytes.")



    # Potencia acumulada de la turbina vs la velocidad del  viento --------------------------------
    cumulative_power = np.array([
        np.trapz(np.nan_to_num(power_densities[:i+1] * weibull_pdf_hm[:i+1]), wind_speeds[:i+1])
        for i in range(len(wind_speeds))
    ])
    # Graficar la potencia acumulada
    plt.figure(figsize=(10, 6))
    plt.plot(wind_speeds, cumulative_power, label='Potencia Acumulada')
    plt.xlabel('Velocidad del Viento (m/s)')
    plt.ylabel('Potencia Acumulada (W)')
    plt.title('Potencia Acumulada de la Turbina vs Velocidad del Viento')
    plt.legend()
    plt.grid(True)
    # Calcular las métricas de potencia
    potencia_maxima = np.max(np.nan_to_num(power_densities*weibull_pdf_hm))
    potencia_promedio = np.mean(np.nan_to_num(power_densities*weibull_pdf_hm))
    # Imprimir los resultados
    # print(f"Potencia máxima: {potencia_maxima:.2f} W")
    # print(f"Potencia promedio: {potencia_promedio:.2f} W")
    # Guardar la gráfica en memoria como PNG
    windturbine_power_cumulative = io.BytesIO()
    plt.savefig(windturbine_power_cumulative, format='png', bbox_inches='tight')  # Guardar como PNG
    plt.close()  # Cerrar la gráfica para liberar memoria
    windturbine_power_cumulative.seek(0)  # Reiniciar el puntero del flujo de datos
    # print(f"La imagen está almacenada en la variable 'windturbine_power_cumulative' y tiene {len(windturbine_power_cumulative.getvalue())} bytes.")

    params_10m_list = list(params_10m)
    params_100m_list = list(params_100m)
    parametrosWeibull_list = list(parametrosWeibull)
    coefficients_list = coefficients.tolist()
    return {
        'datos': {
            'Elevation':response.Elevation(),
            'air_density': air_density
        },
        'Series Temporales': wind_speed_timeseries_bytes,
        'Distribucion Velocidad':wind_speed_distribution,
        'Distribucion Acumulada': wind_distribucion_acomulada,
        'Roseta 10m':wind_rose_10m, 
        'Roseta 100m':wind_rose_100m, 
        'Curva Weibull': {
            'Grafica': Weibull_curves ,
            'Parametros 10m':  params_10m_list,
            'Parametros 100m':  params_100m_list,
        } ,
        'Distribucion Weibull':{
            'Grafica': weibull_distribution,
            'Parametros': parametrosWeibull_list
        },
        'Eficiencia Generador':{
            'Grafica': eficiencia_aerogenerador,
            'Coeficientes':coefficients_list
        },
        'Potencia Viento': potencia_viento,
        'Potencia Turbina':windturbine_power,
        'Potencia Acumulada':{
            'Grafica': windturbine_power_cumulative,
            'Max':potencia_maxima,
            'Promedio':potencia_promedio,
        }
    }

# resultados = wind_analysis(lat=5.69188, lon=-76.65835, start_date="2023-01-01", end_date="2023-12-31")
# def print_resultados(resultados):
    print("\n" + "=" * 50 + "\nRESULTADOS DEL ANÁLISIS\n" + "=" * 50 + "\n")

    # Datos del Lugar
    print("**Datos del Lugar**")
    for key, value in resultados['datos'].items():
        print(f"{key}: {value}")
    print("\n")

    # Series Temporales
    print("**Series Temporales**")
    print(f"Tamaño de la gráfica: {len(resultados['Series Temporales'].getvalue())} bytes")
    print("\n")

    # Distribución Velocidad
    print("**Distribución de Velocidad del Viento**")
    print(f"Tamaño de la gráfica: {len(resultados['Distribucion Velocidad'].getvalue())} bytes")
    print("\n")

    # Distribución Acumulada
    print("**Distribución Acumulada de Velocidad del Viento**")
    print(f"Tamaño de la gráfica: {len(resultados['Distribucion Acumulada'].getvalue())} bytes")
    print("\n")

    # Rosetas de Viento
    print("**Rosetas de Viento**")
    print(f"Tamaño de la gráfica (10m): {len(resultados['Roseta 10m'].getvalue())} bytes")
    print(f"Tamaño de la gráfica (100m): {len(resultados['Roseta 100m'].getvalue())} bytes")
    print("\n")

    # Curva Weibull
    print("**Curva Weibull**")
    print(f"Tamaño de la gráfica: {len(resultados['Curva Weibull']['Grafica'].getvalue())} bytes")
    print(f"Parámetros a 10m: {resultados['Curva Weibull']['Parametros 10m']}")
    print(f"Parámetros a 100m: {resultados['Curva Weibull']['Parametros 100m']}")
    print("\n")

    # Distribución Weibull
    print("**Distribución Weibull**")
    print(f"Tamaño de la gráfica: {len(resultados['Distribucion Weibull']['Grafica'].getvalue())} bytes")
    print(f"Parámetros: {resultados['Distribucion Weibull']['Parametros']}")
    print("\n")

    # Eficiencia del Generador
    print("**Eficiencia del Generador**")
    print(f"Tamaño de la gráfica: {len(resultados['Eficiencia Generador']['Grafica'].getvalue())} bytes")
    print(f"Coeficientes del polinomio de eficiencia: {resultados['Eficiencia Generador']['Coeficientes']}")
    print("\n")

    # Potencia del Viento
    print("**Potencia del Viento**")
    print(f"Tamaño de la gráfica: {len(resultados['Potencia Viento'].getvalue())} bytes")
    print("\n")

    # Potencia Generada por la Turbina
    print("**Potencia Generada por la Turbina**")
    print(f"Tamaño de la gráfica: {len(resultados['Potencia Turbina'].getvalue())} bytes")
    print("\n")

    # Potencia Acumulada
    print("**Potencia Acumulada**")
    print(f"Tamaño de la gráfica: {len(resultados['Potencia Acumulada']['Grafica'].getvalue())} bytes")
    print(f"Potencia máxima: {resultados['Potencia Acumulada']['Max']:.2f} W")
    print(f"Potencia promedio: {resultados['Potencia Acumulada']['Promedio']:.2f} W")
    print("\n")

# Llamar a la función para imprimir resultados
#print_resultados(resultados)
