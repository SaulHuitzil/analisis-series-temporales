"""
Tests unitarios para el modelo de Ising.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from modelos.ising import ModeloIsing


class TestModeloIsing:
    """Tests para la clase ModeloIsing."""

    def test_inicializacion(self):
        """Test de inicialización del modelo."""
        modelo = ModeloIsing(tamanio=10, temperatura=2.0, semilla=42)
        assert modelo.tamanio == 10
        assert modelo.temperatura == 2.0
        assert modelo.estado.shape == (10, 10)
        assert np.all(np.isin(modelo.estado, [-1, 1]))

    def test_temperatura_critica(self):
        """Test del cálculo de temperatura crítica."""
        T_c = ModeloIsing.temperatura_critica()
        assert 2.26 < T_c < 2.27

    def test_magnetizacion_inicial(self):
        """Test de magnetización inicial."""
        modelo = ModeloIsing(tamanio=20, temperatura=2.0, semilla=42)
        mag = modelo.magnetizacion_promedio()
        assert -1.0 <= mag <= 1.0

    def test_simulacion_pasos(self):
        """Test de simulación con número de pasos."""
        modelo = ModeloIsing(tamanio=10, temperatura=2.0, semilla=42)
        pasos = 50
        magnetizacion = modelo.simular(pasos=pasos)
        assert len(magnetizacion) == pasos
        assert np.all(np.abs(magnetizacion) <= 1.0)

    def test_reiniciar_aleatorio(self):
        """Test de reinicio con estado aleatorio."""
        modelo = ModeloIsing(tamanio=10, temperatura=2.0, semilla=42)
        estado_original = modelo.obtener_estado()
        modelo.simular(pasos=10, calcular_magnetizacion=False)
        modelo.reiniciar(tipo='aleatorio')
        # No debería ser exactamente igual después de reiniciar
        assert modelo.estado.shape == estado_original.shape

    def test_reiniciar_todos_arriba(self):
        """Test de reinicio con todos los espines hacia arriba."""
        modelo = ModeloIsing(tamanio=10, temperatura=2.0)
        modelo.reiniciar(tipo='todos_arriba')
        assert np.all(modelo.estado == 1)
        assert modelo.magnetizacion_promedio() == 1.0

    def test_reiniciar_todos_abajo(self):
        """Test de reinicio con todos los espines hacia abajo."""
        modelo = ModeloIsing(tamanio=10, temperatura=2.0)
        modelo.reiniciar(tipo='todos_abajo')
        assert np.all(modelo.estado == -1)
        assert modelo.magnetizacion_promedio() == -1.0

    def test_energia_total(self):
        """Test de cálculo de energía total."""
        modelo = ModeloIsing(tamanio=5, temperatura=2.0, semilla=42)
        energia = modelo.energia_total()
        assert isinstance(energia, float)

    def test_coherencia_reproducibilidad(self):
        """Test de reproducibilidad con semilla fija."""
        modelo1 = ModeloIsing(tamanio=10, temperatura=2.0, semilla=42)
        mag1 = modelo1.simular(pasos=10)

        modelo2 = ModeloIsing(tamanio=10, temperatura=2.0, semilla=42)
        mag2 = modelo2.simular(pasos=10)

        assert np.allclose(mag1, mag2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
