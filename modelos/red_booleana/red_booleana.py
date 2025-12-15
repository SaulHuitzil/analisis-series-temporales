"""
Implementación de Redes Booleanas Aleatorias (Random Boolean Networks).

Las redes booleanas son modelos de sistemas dinámicos discretos donde cada nodo
puede estar en estado 0 o 1, y evoluciona según reglas booleanas que dependen
de sus nodos de entrada.

Las redes booleanas presentan tres regímenes dinámicos:
- Ordenado (K < 2): converge a un atractor
- Crítico (K ≈ 2): comportamiento complejo
- Caótico (K > 2): dinámicas impredecibles
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class RedBooleana:
    """
    Implementación de una Red Booleana Aleatoria (RBN).

    Attributes:
        n_nodos: Número de nodos en la red
        k: Número promedio de entradas por nodo
        p_bias: Probabilidad de sesgo hacia 1 en las funciones booleanas
        estado: Vector de estado actual (0s y 1s)
        tabla_conexiones: Matriz que define las conexiones entre nodos
        funciones: Lista de funciones booleanas para cada nodo
    """

    def __init__(self,
                 n_nodos: int = 100,
                 k: int = 2,
                 p_bias: float = 0.5,
                 semilla: Optional[int] = None):
        """
        Inicializa una red booleana aleatoria.

        Args:
            n_nodos: Número de nodos en la red
            k: Número de entradas por nodo (conectividad)
            p_bias: Probabilidad de que la función booleana retorne 1
            semilla: Semilla para el generador aleatorio
        """
        self.n_nodos = n_nodos
        self.k = k
        self.p_bias = p_bias

        if semilla is not None:
            np.random.seed(semilla)

        # Inicializar estado aleatorio
        self.estado = np.random.randint(0, 2, size=n_nodos)

        # Generar tabla de conexiones
        self.tabla_conexiones = self._generar_conexiones()

        # Generar funciones booleanas aleatorias
        self.funciones = self._generar_funciones()

        # Historial para detectar atractores
        self.historial = []

    def _generar_conexiones(self) -> np.ndarray:
        """
        Genera la tabla de conexiones de la red.

        Returns:
            Matriz (n_nodos x k) con los índices de nodos de entrada para cada nodo
        """
        conexiones = np.zeros((self.n_nodos, self.k), dtype=int)

        for i in range(self.n_nodos):
            # Seleccionar k nodos de entrada aleatoriamente
            conexiones[i] = np.random.choice(self.n_nodos, size=self.k, replace=True)

        return conexiones

    def _generar_funciones(self) -> List[np.ndarray]:
        """
        Genera funciones booleanas aleatorias para cada nodo.

        Cada función es una tabla de verdad con 2^k entradas.

        Returns:
            Lista de tablas de verdad (una por nodo)
        """
        funciones = []

        for _ in range(self.n_nodos):
            # Generar tabla de verdad aleatoria
            n_entradas = 2 ** self.k
            tabla_verdad = (np.random.random(n_entradas) < self.p_bias).astype(int)
            funciones.append(tabla_verdad)

        return funciones

    def _evaluar_nodo(self, indice_nodo: int) -> int:
        """
        Evalúa la función booleana de un nodo dado su estado de entrada.

        Args:
            indice_nodo: Índice del nodo a evaluar

        Returns:
            Nuevo estado del nodo (0 o 1)
        """
        # Obtener los índices de los nodos de entrada
        indices_entrada = self.tabla_conexiones[indice_nodo]

        # Obtener los valores de los nodos de entrada
        valores_entrada = self.estado[indices_entrada]

        # Convertir los valores binarios a un índice decimal
        indice = sum(valor * (2 ** i) for i, valor in enumerate(valores_entrada))

        # Evaluar la función booleana
        return self.funciones[indice_nodo][indice]

    def paso(self, modo: str = 'sincrono'):
        """
        Realiza un paso de evolución de la red.

        Args:
            modo: Modo de actualización ('sincrono' o 'asincrono')
        """
        if modo == 'sincrono':
            # Actualización síncrona: todos los nodos se actualizan simultáneamente
            nuevo_estado = np.array([self._evaluar_nodo(i) for i in range(self.n_nodos)])
            self.estado = nuevo_estado

        elif modo == 'asincrono':
            # Actualización asíncrona: actualizar un nodo aleatorio
            nodo = np.random.randint(0, self.n_nodos)
            self.estado[nodo] = self._evaluar_nodo(nodo)

        else:
            raise ValueError(f"Modo desconocido: {modo}. Use 'sincrono' o 'asincrono'.")

    def simular(self,
                pasos: int,
                modo: str = 'sincrono',
                guardar_historial: bool = False) -> np.ndarray:
        """
        Simula la red durante un número dado de pasos.

        Args:
            pasos: Número de pasos de evolución
            modo: Modo de actualización ('sincrono' o 'asincrono')
            guardar_historial: Si True, guarda todos los estados

        Returns:
            Array con el estado promedio (fracción de nodos en 1) en cada paso
        """
        estados_promedio = np.zeros(pasos)

        if guardar_historial:
            self.historial = [self.estado.copy()]

        for paso in range(pasos):
            self.paso(modo)
            estados_promedio[paso] = self.estado_promedio()

            if guardar_historial:
                self.historial.append(self.estado.copy())

        return estados_promedio

    def estado_promedio(self) -> float:
        """
        Calcula el estado promedio de la red.

        Returns:
            Fracción de nodos en estado 1 (entre 0 y 1)
        """
        return np.mean(self.estado)

    def detectar_atractor(self, max_pasos: int = 1000) -> Tuple[int, int]:
        """
        Detecta si la red ha alcanzado un atractor.

        Args:
            max_pasos: Número máximo de pasos para buscar el atractor

        Returns:
            Tupla (periodo, transiente) del atractor encontrado.
            periodo=1 indica punto fijo, periodo>1 indica ciclo límite.
        """
        estados_visitados = {}
        self.historial = []

        for paso in range(max_pasos):
            # Convertir estado a tupla para usar como key
            estado_tuple = tuple(self.estado)

            if estado_tuple in estados_visitados:
                # Encontramos un ciclo
                transiente = estados_visitados[estado_tuple]
                periodo = paso - transiente
                return periodo, transiente

            estados_visitados[estado_tuple] = paso
            self.historial.append(self.estado.copy())
            self.paso()

        # No se encontró atractor
        return -1, -1

    def distancia_hamming(self, otro_estado: np.ndarray) -> int:
        """
        Calcula la distancia de Hamming entre el estado actual y otro estado.

        Args:
            otro_estado: Array de estado para comparar

        Returns:
            Distancia de Hamming (número de bits diferentes)
        """
        return np.sum(self.estado != otro_estado)

    def reiniciar(self, tipo: str = 'aleatorio', estado_inicial: Optional[np.ndarray] = None):
        """
        Reinicia el estado de la red.

        Args:
            tipo: Tipo de inicialización ('aleatorio', 'todos_ceros', 'todos_unos', 'personalizado')
            estado_inicial: Estado inicial personalizado (si tipo='personalizado')
        """
        if tipo == 'aleatorio':
            self.estado = np.random.randint(0, 2, size=self.n_nodos)
        elif tipo == 'todos_ceros':
            self.estado = np.zeros(self.n_nodos, dtype=int)
        elif tipo == 'todos_unos':
            self.estado = np.ones(self.n_nodos, dtype=int)
        elif tipo == 'personalizado':
            if estado_inicial is None:
                raise ValueError("Debe proporcionar estado_inicial para tipo='personalizado'")
            if len(estado_inicial) != self.n_nodos:
                raise ValueError(f"estado_inicial debe tener {self.n_nodos} elementos")
            self.estado = estado_inicial.copy()
        else:
            raise ValueError(f"Tipo de inicialización desconocido: {tipo}")

        self.historial = []

    def obtener_estado(self) -> np.ndarray:
        """
        Retorna una copia del estado actual.

        Returns:
            Copia del vector de estado
        """
        return self.estado.copy()

    @staticmethod
    def conectividad_critica() -> int:
        """
        Retorna la conectividad crítica teórica para p_bias=0.5.

        Returns:
            K crítico = 2
        """
        return 2
