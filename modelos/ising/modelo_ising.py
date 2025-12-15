"""
Implementación del modelo de Ising 2D.

El modelo de Ising es un modelo matemático de ferromagnetismo en mecánica
estadística. Consiste en espines discretos dispuestos en una red que pueden
estar en dos estados (+1 o -1).
"""

import numpy as np
from typing import Tuple, Optional


class ModeloIsing:
    """
    Implementación del modelo de Ising 2D con algoritmo de Metrópolis.

    El modelo de Ising muestra una transición de fase a temperatura crítica Tc.
    Para una red 2D cuadrada, Tc ≈ 2.269 J/k_B.

    Attributes:
        tamanio: Tamaño de la red (tamanio x tamanio)
        temperatura: Temperatura del sistema (en unidades de J/k_B)
        campo_externo: Campo magnético externo
        estado: Matriz de espines (-1 o +1)
    """

    def __init__(self,
                 tamanio: int = 50,
                 temperatura: float = 2.269,
                 campo_externo: float = 0.0,
                 semilla: Optional[int] = None):
        """
        Inicializa el modelo de Ising.

        Args:
            tamanio: Tamaño de la red cuadrada
            temperatura: Temperatura en unidades de J/k_B
            campo_externo: Campo magnético externo
            semilla: Semilla para el generador aleatorio
        """
        self.tamanio = tamanio
        self.temperatura = temperatura
        self.campo_externo = campo_externo

        if semilla is not None:
            np.random.seed(semilla)

        # Inicializar espines aleatoriamente
        self.estado = np.random.choice([-1, 1], size=(tamanio, tamanio))

        # Pre-calcular probabilidades de transición para eficiencia
        self._precalcular_probabilidades()

    def _precalcular_probabilidades(self):
        """Pre-calcula las probabilidades de transición del algoritmo de Metrópolis."""
        self.probabilidades = {}
        for dE in range(-8, 9, 2):  # Posibles cambios de energía: -8, -6, ..., 6, 8
            self.probabilidades[dE] = np.exp(-dE / self.temperatura)

    def _energia_local(self, i: int, j: int) -> float:
        """
        Calcula la energía local de un espín en la posición (i, j).

        Args:
            i: Índice de fila
            j: Índice de columna

        Returns:
            Energía local del espín
        """
        # Condiciones de frontera periódicas
        espin = self.estado[i, j]
        vecinos = (
            self.estado[(i + 1) % self.tamanio, j] +
            self.estado[(i - 1) % self.tamanio, j] +
            self.estado[i, (j + 1) % self.tamanio] +
            self.estado[i, (j - 1) % self.tamanio]
        )

        return -espin * (vecinos + self.campo_externo)

    def paso_montecarlo(self):
        """
        Realiza un paso de Monte Carlo (un barrido completo de la red).

        En cada paso, se intenta voltear cada espín una vez en promedio
        usando el algoritmo de Metrópolis.
        """
        for _ in range(self.tamanio * self.tamanio):
            # Seleccionar un espín aleatorio
            i = np.random.randint(0, self.tamanio)
            j = np.random.randint(0, self.tamanio)

            # Calcular cambio de energía si se voltea el espín
            dE = -2 * self._energia_local(i, j)

            # Criterio de Metrópolis
            if dE <= 0 or np.random.random() < self.probabilidades[int(dE)]:
                self.estado[i, j] *= -1

    def simular(self, pasos: int, calcular_magnetizacion: bool = True) -> np.ndarray:
        """
        Simula el modelo de Ising durante un número dado de pasos.

        Args:
            pasos: Número de pasos de Monte Carlo
            calcular_magnetizacion: Si True, retorna la magnetización en cada paso

        Returns:
            Array con la magnetización promedio en cada paso (si calcular_magnetizacion=True)
        """
        if calcular_magnetizacion:
            magnetizaciones = np.zeros(pasos)

            for paso in range(pasos):
                self.paso_montecarlo()
                magnetizaciones[paso] = self.magnetizacion_promedio()

            return magnetizaciones
        else:
            for _ in range(pasos):
                self.paso_montecarlo()
            return np.array([])

    def magnetizacion_promedio(self) -> float:
        """
        Calcula la magnetización promedio del sistema.

        Returns:
            Magnetización promedio (entre -1 y 1)
        """
        return np.mean(self.estado)

    def energia_total(self) -> float:
        """
        Calcula la energía total del sistema.

        Returns:
            Energía total
        """
        energia = 0.0

        for i in range(self.tamanio):
            for j in range(self.tamanio):
                # Solo contar cada par una vez
                espin = self.estado[i, j]
                vecino_derecha = self.estado[i, (j + 1) % self.tamanio]
                vecino_abajo = self.estado[(i + 1) % self.tamanio, j]

                energia += -espin * (vecino_derecha + vecino_abajo)

                # Campo externo
                energia += -espin * self.campo_externo

        return energia

    def reiniciar(self, tipo: str = 'aleatorio'):
        """
        Reinicia el estado del sistema.

        Args:
            tipo: Tipo de inicialización ('aleatorio', 'todos_arriba', 'todos_abajo')
        """
        if tipo == 'aleatorio':
            self.estado = np.random.choice([-1, 1], size=(self.tamanio, self.tamanio))
        elif tipo == 'todos_arriba':
            self.estado = np.ones((self.tamanio, self.tamanio), dtype=int)
        elif tipo == 'todos_abajo':
            self.estado = -np.ones((self.tamanio, self.tamanio), dtype=int)
        else:
            raise ValueError(f"Tipo de inicialización desconocido: {tipo}")

    def obtener_estado(self) -> np.ndarray:
        """
        Retorna una copia del estado actual del sistema.

        Returns:
            Copia de la matriz de espines
        """
        return self.estado.copy()

    @staticmethod
    def temperatura_critica() -> float:
        """
        Retorna la temperatura crítica teórica del modelo de Ising 2D.

        Returns:
            Temperatura crítica (Tc ≈ 2.269 J/k_B)
        """
        return 2.0 / np.log(1.0 + np.sqrt(2.0))
