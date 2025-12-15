"""
Tests unitarios para las redes booleanas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from modelos.red_booleana import RedBooleana


class TestRedBooleana:
    """Tests para la clase RedBooleana."""

    def test_inicializacion(self):
        """Test de inicialización de la red."""
        red = RedBooleana(n_nodos=50, k=2, p_bias=0.5, semilla=42)
        assert red.n_nodos == 50
        assert red.k == 2
        assert red.p_bias == 0.5
        assert len(red.estado) == 50
        assert np.all(np.isin(red.estado, [0, 1]))

    def test_conectividad_critica(self):
        """Test del valor de conectividad crítica."""
        K_c = RedBooleana.conectividad_critica()
        assert K_c == 2

    def test_estado_promedio(self):
        """Test de cálculo de estado promedio."""
        red = RedBooleana(n_nodos=100, k=2, semilla=42)
        estado_prom = red.estado_promedio()
        assert 0.0 <= estado_prom <= 1.0

    def test_simulacion_sincrona(self):
        """Test de simulación síncrona."""
        red = RedBooleana(n_nodos=50, k=2, semilla=42)
        pasos = 100
        estados = red.simular(pasos=pasos, modo='sincrono')
        assert len(estados) == pasos
        assert np.all((estados >= 0) & (estados <= 1))

    def test_simulacion_asincrona(self):
        """Test de simulación asíncrona."""
        red = RedBooleana(n_nodos=50, k=2, semilla=42)
        pasos = 100
        estados = red.simular(pasos=pasos, modo='asincrono')
        assert len(estados) == pasos
        assert np.all((estados >= 0) & (estados <= 1))

    def test_reiniciar_aleatorio(self):
        """Test de reinicio con estado aleatorio."""
        red = RedBooleana(n_nodos=50, k=2, semilla=42)
        red.simular(pasos=10)
        red.reiniciar(tipo='aleatorio')
        assert len(red.estado) == 50
        assert np.all(np.isin(red.estado, [0, 1]))

    def test_reiniciar_todos_ceros(self):
        """Test de reinicio con todos los nodos en 0."""
        red = RedBooleana(n_nodos=50, k=2)
        red.reiniciar(tipo='todos_ceros')
        assert np.all(red.estado == 0)
        assert red.estado_promedio() == 0.0

    def test_reiniciar_todos_unos(self):
        """Test de reinicio con todos los nodos en 1."""
        red = RedBooleana(n_nodos=50, k=2)
        red.reiniciar(tipo='todos_unos')
        assert np.all(red.estado == 1)
        assert red.estado_promedio() == 1.0

    def test_reiniciar_personalizado(self):
        """Test de reinicio con estado personalizado."""
        red = RedBooleana(n_nodos=10, k=2)
        estado_custom = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        red.reiniciar(tipo='personalizado', estado_inicial=estado_custom)
        assert np.array_equal(red.estado, estado_custom)

    def test_distancia_hamming(self):
        """Test de cálculo de distancia de Hamming."""
        red = RedBooleana(n_nodos=10, k=2)
        red.reiniciar(tipo='todos_ceros')
        otro_estado = np.ones(10, dtype=int)
        distancia = red.distancia_hamming(otro_estado)
        assert distancia == 10

    def test_detectar_atractor_punto_fijo(self):
        """Test de detección de punto fijo (K=1 tiende a puntos fijos)."""
        red = RedBooleana(n_nodos=20, k=1, semilla=42)
        periodo, transiente = red.detectar_atractor(max_pasos=200)
        # Para K=1, es muy probable encontrar un atractor
        assert periodo >= 0
        assert transiente >= 0

    def test_guardar_historial(self):
        """Test de guardado de historial durante simulación."""
        red = RedBooleana(n_nodos=20, k=2, semilla=42)
        pasos = 50
        red.simular(pasos=pasos, guardar_historial=True)
        # Historial debe tener pasos + 1 (incluye estado inicial)
        assert len(red.historial) == pasos + 1

    def test_coherencia_reproducibilidad(self):
        """Test de reproducibilidad con semilla fija."""
        red1 = RedBooleana(n_nodos=50, k=2, semilla=42)
        estados1 = red1.simular(pasos=20)

        red2 = RedBooleana(n_nodos=50, k=2, semilla=42)
        estados2 = red2.simular(pasos=20)

        assert np.allclose(estados1, estados2)

    def test_tabla_conexiones_dimensiones(self):
        """Test de dimensiones de la tabla de conexiones."""
        red = RedBooleana(n_nodos=30, k=3, semilla=42)
        assert red.tabla_conexiones.shape == (30, 3)

    def test_funciones_booleanas_longitud(self):
        """Test de longitud de las funciones booleanas."""
        red = RedBooleana(n_nodos=10, k=2, semilla=42)
        assert len(red.funciones) == 10
        for func in red.funciones:
            assert len(func) == 2**2  # 2^k entradas


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
