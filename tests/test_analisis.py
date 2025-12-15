"""
Tests unitarios para el módulo de análisis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from src.analisis import (
    calcular_variabilidad,
    calcular_variabilidad_ventana,
    calcular_autocorrelacion,
    detectar_criticalidad,
    calcular_entropias,
    calcular_metricas_completas
)


class TestAnalisis:
    """Tests para las funciones de análisis."""

    def test_calcular_variabilidad_constante(self):
        """Test de variabilidad de serie constante."""
        serie = np.ones(100)
        var = calcular_variabilidad(serie)
        assert var == 0.0

    def test_calcular_variabilidad_variable(self):
        """Test de variabilidad de serie variable."""
        serie = np.random.randn(1000)
        var = calcular_variabilidad(serie)
        assert var > 0.0

    def test_calcular_variabilidad_ventana(self):
        """Test de variabilidad en ventanas."""
        serie = np.random.randn(100)
        ventanas = calcular_variabilidad_ventana(serie, tamanio_ventana=10)
        assert len(ventanas) == 100 - 10 + 1
        assert np.all(ventanas >= 0)

    def test_calcular_autocorrelacion(self):
        """Test de cálculo de autocorrelación."""
        serie = np.random.randn(200)
        autocorr = calcular_autocorrelacion(serie, max_lag=50)
        assert len(autocorr) == 50
        assert autocorr[0] == 1.0  # Autocorrelación en lag=0 debe ser 1
        assert np.all(np.abs(autocorr) <= 1.0)

    def test_detectar_criticalidad_alta_variabilidad(self):
        """Test de detección de criticalidad con alta variabilidad."""
        # Crear serie con alta variabilidad
        serie = np.random.uniform(-1, 1, 500)
        resultado = detectar_criticalidad(serie, umbral_variabilidad=0.1)
        assert 'variabilidad' in resultado
        assert 'tiempo_correlacion' in resultado
        assert 'es_critico' in resultado
        assert isinstance(resultado['es_critico'], bool)

    def test_detectar_criticalidad_baja_variabilidad(self):
        """Test de detección con baja variabilidad."""
        # Crear serie con baja variabilidad
        serie = np.random.uniform(0.49, 0.51, 500)
        resultado = detectar_criticalidad(serie, umbral_variabilidad=0.1)
        assert resultado['variabilidad'] < 0.1

    def test_calcular_entropias(self):
        """Test de cálculo de entropía."""
        serie_uniforme = np.random.uniform(0, 1, 1000)
        entropia = calcular_entropias(serie_uniforme, bins=20)
        assert entropia > 0
        assert isinstance(entropia, float)

    def test_calcular_entropias_constante(self):
        """Test de entropía de serie constante."""
        serie_constante = np.ones(100)
        entropia = calcular_entropias(serie_constante, bins=20)
        # Serie constante debería tener entropía muy baja
        assert entropia >= 0

    def test_calcular_metricas_completas(self):
        """Test de cálculo de métricas completas."""
        serie = np.random.randn(500)
        metricas = calcular_metricas_completas(serie)

        # Verificar que todas las claves estén presentes
        claves_esperadas = [
            'media', 'varianza', 'desviacion_estandar',
            'minimo', 'maximo', 'rango', 'entropia',
            'variabilidad', 'tiempo_correlacion', 'es_critico'
        ]
        for clave in claves_esperadas:
            assert clave in metricas

        # Verificar tipos y valores razonables
        assert isinstance(metricas['media'], float)
        assert metricas['varianza'] >= 0
        assert metricas['desviacion_estandar'] >= 0
        assert metricas['rango'] >= 0
        assert metricas['entropia'] >= 0

    def test_metricas_serie_conocida(self):
        """Test de métricas con serie conocida."""
        serie = np.array([1, 2, 3, 4, 5])
        metricas = calcular_metricas_completas(serie)

        assert metricas['media'] == 3.0
        assert metricas['minimo'] == 1
        assert metricas['maximo'] == 5
        assert metricas['rango'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
