"""
Ejemplo de comparación de comportamiento crítico vs no-crítico.

Este script simula tanto el modelo de Ising como redes booleanas en
regímenes críticos y no críticos, calcula la variabilidad del estado
promedio en 1000 pasos, y genera gráficos comparativos.
"""

import sys
import os

# Añadir el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from modelos.ising import ModeloIsing
from modelos.red_booleana import RedBooleana
from src.analisis import calcular_metricas_completas
from src.visualizacion import graficar_variabilidad_comparativa


def simular_ising_critico_vs_no_critico(pasos: int = 1000, tamanio: int = 50):
    """
    Simula el modelo de Ising en régimen crítico y no crítico.

    Args:
        pasos: Número de pasos de Monte Carlo
        tamanio: Tamaño de la red

    Returns:
        Diccionario con las series temporales
    """
    print("\n=== Simulando Modelo de Ising ===")

    # Temperatura crítica del modelo de Ising 2D
    T_c = ModeloIsing.temperatura_critica()
    print(f"Temperatura crítica teórica: T_c = {T_c:.4f}")

    # 1. Régimen crítico (T ≈ T_c)
    print(f"\n1. Simulando en régimen crítico (T = {T_c:.4f})...")
    ising_critico = ModeloIsing(tamanio=tamanio, temperatura=T_c, semilla=42)
    # Termalización: dejar que el sistema se equilibre
    print("   Termalizando...")
    ising_critico.simular(pasos=200, calcular_magnetizacion=False)
    print("   Simulando...")
    magnetizacion_critica = ising_critico.simular(pasos=pasos)

    # 2. Régimen ordenado (T < T_c)
    T_ordenado = T_c * 0.6
    print(f"\n2. Simulando en régimen ordenado (T = {T_ordenado:.4f})...")
    ising_ordenado = ModeloIsing(tamanio=tamanio, temperatura=T_ordenado, semilla=42)
    print("   Termalizando...")
    ising_ordenado.simular(pasos=200, calcular_magnetizacion=False)
    print("   Simulando...")
    magnetizacion_ordenada = ising_ordenado.simular(pasos=pasos)

    # 3. Régimen caótico/desordenado (T > T_c)
    T_caotico = T_c * 1.5
    print(f"\n3. Simulando en régimen desordenado (T = {T_caotico:.4f})...")
    ising_caotico = ModeloIsing(tamanio=tamanio, temperatura=T_caotico, semilla=42)
    print("   Termalizando...")
    ising_caotico.simular(pasos=200, calcular_magnetizacion=False)
    print("   Simulando...")
    magnetizacion_caotica = ising_caotico.simular(pasos=pasos)

    return {
        'Ising Crítico (T≈Tc)': np.abs(magnetizacion_critica),
        'Ising Ordenado (T<Tc)': np.abs(magnetizacion_ordenada),
        'Ising Desordenado (T>Tc)': np.abs(magnetizacion_caotica)
    }


def simular_rbn_critico_vs_no_critico(pasos: int = 1000, n_nodos: int = 100):
    """
    Simula redes booleanas en régimen crítico y no crítico.

    Args:
        pasos: Número de pasos de evolución
        n_nodos: Número de nodos en la red

    Returns:
        Diccionario con las series temporales
    """
    print("\n=== Simulando Redes Booleanas ===")

    K_c = RedBooleana.conectividad_critica()
    print(f"Conectividad crítica teórica: K_c = {K_c}")

    # 1. Régimen crítico (K = K_c = 2)
    print(f"\n1. Simulando en régimen crítico (K = {K_c})...")
    rbn_critica = RedBooleana(n_nodos=n_nodos, k=K_c, p_bias=0.5, semilla=42)
    estados_criticos = rbn_critica.simular(pasos=pasos)

    # 2. Régimen ordenado (K < K_c)
    K_ordenado = 1
    print(f"\n2. Simulando en régimen ordenado (K = {K_ordenado})...")
    rbn_ordenada = RedBooleana(n_nodos=n_nodos, k=K_ordenado, p_bias=0.5, semilla=42)
    estados_ordenados = rbn_ordenada.simular(pasos=pasos)

    # 3. Régimen caótico (K > K_c)
    K_caotico = 3
    print(f"\n3. Simulando en régimen caótico (K = {K_caotico})...")
    rbn_caotica = RedBooleana(n_nodos=n_nodos, k=K_caotico, p_bias=0.5, semilla=42)
    estados_caoticos = rbn_caotica.simular(pasos=pasos)

    return {
        'RBN Crítica (K=2)': estados_criticos,
        'RBN Ordenada (K=1)': estados_ordenados,
        'RBN Caótica (K=3)': estados_caoticos
    }


def main():
    """Función principal del ejemplo."""
    print("=" * 70)
    print("ANÁLISIS DE VARIABILIDAD TEMPORAL EN SISTEMAS CRÍTICOS")
    print("=" * 70)

    # Parámetros de simulación
    pasos = 1000
    print(f"\nParámetros:")
    print(f"  - Pasos de simulación: {pasos}")
    print(f"  - Tamaño de red Ising: 50x50")
    print(f"  - Número de nodos RBN: 100")

    # 1. Simular modelo de Ising
    series_ising = simular_ising_critico_vs_no_critico(pasos=pasos)

    # 2. Simular redes booleanas
    series_rbn = simular_rbn_critico_vs_no_critico(pasos=pasos)

    # 3. Análisis de métricas
    print("\n" + "=" * 70)
    print("ANÁLISIS DE MÉTRICAS")
    print("=" * 70)

    print("\n--- Modelo de Ising ---")
    for nombre, serie in series_ising.items():
        metricas = calcular_metricas_completas(serie)
        print(f"\n{nombre}:")
        print(f"  Variabilidad (std): {metricas['desviacion_estandar']:.6f}")
        print(f"  Media: {metricas['media']:.6f}")
        print(f"  Entropía: {metricas['entropia']:.6f}")
        print(f"  Tiempo de correlación: {metricas['tiempo_correlacion']}")
        print(f"  ¿Es crítico?: {metricas['es_critico']}")

    print("\n--- Redes Booleanas ---")
    for nombre, serie in series_rbn.items():
        metricas = calcular_metricas_completas(serie)
        print(f"\n{nombre}:")
        print(f"  Variabilidad (std): {metricas['desviacion_estandar']:.6f}")
        print(f"  Media: {metricas['media']:.6f}")
        print(f"  Entropía: {metricas['entropia']:.6f}")
        print(f"  Tiempo de correlación: {metricas['tiempo_correlacion']}")
        print(f"  ¿Es crítico?: {metricas['es_critico']}")

    # 4. Visualización
    print("\n" + "=" * 70)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 70)

    # Gráfico comparativo para Ising
    print("\nGenerando gráfico comparativo del modelo de Ising...")
    fig1 = graficar_variabilidad_comparativa(series_ising)
    fig1.suptitle('Análisis de Variabilidad - Modelo de Ising',
                  fontsize=16, fontweight='bold', y=1.00)
    plt.savefig('examples/comparacion_ising.png', dpi=300, bbox_inches='tight')
    print("  Guardado en: examples/comparacion_ising.png")

    # Gráfico comparativo para RBN
    print("\nGenerando gráfico comparativo de Redes Booleanas...")
    fig2 = graficar_variabilidad_comparativa(series_rbn)
    fig2.suptitle('Análisis de Variabilidad - Redes Booleanas',
                  fontsize=16, fontweight='bold', y=1.00)
    plt.savefig('examples/comparacion_rbn.png', dpi=300, bbox_inches='tight')
    print("  Guardado en: examples/comparacion_rbn.png")

    # Gráfico combinado
    print("\nGenerando gráfico comparativo combinado...")
    todas_series = {**series_ising, **series_rbn}
    fig3 = graficar_variabilidad_comparativa(todas_series)
    fig3.suptitle('Análisis Comparativo - Todos los Modelos',
                  fontsize=16, fontweight='bold', y=1.00)
    plt.savefig('examples/comparacion_completa.png', dpi=300, bbox_inches='tight')
    print("  Guardado en: examples/comparacion_completa.png")

    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)
    print("""
Los sistemas críticos presentan características distintivas:

1. VARIABILIDAD INTERMEDIA:
   - Los sistemas críticos muestran variabilidad entre los regímenes
     ordenado y caótico.
   - Esto indica un balance entre orden y desorden.

2. LARGO TIEMPO DE CORRELACIÓN:
   - Las correlaciones temporales persisten más tiempo en el punto crítico.
   - Esto sugiere memoria a largo plazo en el sistema.

3. MAYOR ENTROPÍA:
   - Los sistemas críticos exploran un mayor espacio de estados.
   - Esto se relaciona con su capacidad de procesamiento de información.

4. TRANSICIONES DE FASE:
   - El modelo de Ising muestra transición de fase en T_c.
   - Las RBN muestran transición en K_c = 2.

Estos resultados son consistentes con la teoría de sistemas críticos
y su relevancia en sistemas biológicos y físicos.
    """)

    print("\n✓ Análisis completado exitosamente.")


if __name__ == "__main__":
    main()
