"""
Módulo de análisis de variabilidad temporal.

Este módulo proporciona herramientas para analizar la variabilidad
temporal en sistemas críticos.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


def calcular_variabilidad(serie_temporal: np.ndarray) -> float:
    """
    Calcula la variabilidad de una serie temporal usando desviación estándar.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal

    Returns:
        Desviación estándar de la serie
    """
    return np.std(serie_temporal)


def calcular_variabilidad_ventana(serie_temporal: np.ndarray,
                                   tamanio_ventana: int = 10) -> np.ndarray:
    """
    Calcula la variabilidad en ventanas deslizantes.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal
        tamanio_ventana: Tamaño de la ventana deslizante

    Returns:
        Array con la variabilidad en cada ventana
    """
    n = len(serie_temporal)
    variabilidades = []

    for i in range(n - tamanio_ventana + 1):
        ventana = serie_temporal[i:i + tamanio_ventana]
        variabilidades.append(np.std(ventana))

    return np.array(variabilidades)


def calcular_autocorrelacion(serie_temporal: np.ndarray,
                            max_lag: int = 50) -> np.ndarray:
    """
    Calcula la función de autocorrelación de una serie temporal.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal
        max_lag: Número máximo de lags a calcular

    Returns:
        Array con los valores de autocorrelación para cada lag
    """
    serie_normalizada = serie_temporal - np.mean(serie_temporal)
    autocorr = np.correlate(serie_normalizada, serie_normalizada, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    return autocorr[:max_lag]


def detectar_criticalidad(serie_temporal: np.ndarray,
                         umbral_variabilidad: float = 0.1) -> Dict[str, float]:
    """
    Analiza si un sistema presenta comportamiento crítico.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal
        umbral_variabilidad: Umbral para considerar alta variabilidad

    Returns:
        Diccionario con métricas de criticalidad
    """
    variabilidad = calcular_variabilidad(serie_temporal)
    autocorr = calcular_autocorrelacion(serie_temporal)

    # Tiempo de correlación (primera vez que la autocorrelación cae bajo 1/e)
    umbral_autocorr = 1.0 / np.e
    tiempo_correlacion = np.argmax(autocorr < umbral_autocorr)
    if tiempo_correlacion == 0 and autocorr[0] >= umbral_autocorr:
        tiempo_correlacion = len(autocorr)

    return {
        'variabilidad': variabilidad,
        'tiempo_correlacion': tiempo_correlacion,
        'es_critico': variabilidad > umbral_variabilidad and tiempo_correlacion > 5
    }


def calcular_entropias(serie_temporal: np.ndarray, bins: int = 20) -> float:
    """
    Calcula la entropía de Shannon de una serie temporal.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal
        bins: Número de bins para discretizar la serie

    Returns:
        Entropía de Shannon
    """
    hist, _ = np.histogram(serie_temporal, bins=bins, density=True)
    hist = hist[hist > 0]  # Eliminar bins vacíos
    entropia = -np.sum(hist * np.log2(hist + 1e-10))

    return entropia


def calcular_metricas_completas(serie_temporal: np.ndarray) -> Dict[str, float]:
    """
    Calcula un conjunto completo de métricas para una serie temporal.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal

    Returns:
        Diccionario con todas las métricas calculadas
    """
    return {
        'media': np.mean(serie_temporal),
        'varianza': np.var(serie_temporal),
        'desviacion_estandar': np.std(serie_temporal),
        'minimo': np.min(serie_temporal),
        'maximo': np.max(serie_temporal),
        'rango': np.max(serie_temporal) - np.min(serie_temporal),
        'entropia': calcular_entropias(serie_temporal),
        **detectar_criticalidad(serie_temporal)
    }
