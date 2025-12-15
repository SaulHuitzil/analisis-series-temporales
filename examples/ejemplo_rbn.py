"""
Ejemplo básico de Redes Booleanas.

Este script demuestra el uso básico de las redes booleanas,
incluyendo simulación y detección de atractores.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from modelos.red_booleana import RedBooleana
from src.visualizacion import graficar_serie_temporal


def main():
    """Función principal del ejemplo."""
    print("=" * 60)
    print("EJEMPLO: REDES BOOLEANAS ALEATORIAS")
    print("=" * 60)

    # Parámetros
    n_nodos = 100
    K_c = RedBooleana.conectividad_critica()

    print(f"\nParámetros:")
    print(f"  Número de nodos: {n_nodos}")
    print(f"  Conectividad crítica: K_c = {K_c}")

    # Crear red en régimen crítico
    print(f"\nCreando red booleana con K = {K_c}...")
    red = RedBooleana(n_nodos=n_nodos, k=K_c, p_bias=0.5, semilla=42)

    print(f"Estado inicial promedio: {red.estado_promedio():.4f}")

    # Simular
    print("\nSimulando 1000 pasos...")
    estados = red.simular(pasos=1000, guardar_historial=True)

    # Graficar evolución
    print("\nGraficando evolución del estado promedio...")
    fig1, ax = plt.subplots(figsize=(10, 4))
    graficar_serie_temporal(estados,
                           titulo="Evolución del Estado Promedio (RBN, K=2)",
                           ylabel="Fracción de Nodos Activos",
                           color='darkgreen',
                           ax=ax)
    plt.savefig('examples/rbn_evolucion.png', dpi=150, bbox_inches='tight')
    print("  Guardado en: examples/rbn_evolucion.png")

    # Estadísticas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS")
    print("=" * 60)
    print(f"Estado promedio: {np.mean(estados):.6f}")
    print(f"Desviación estándar: {np.std(estados):.6f}")
    print(f"Estado mínimo: {np.min(estados):.6f}")
    print(f"Estado máximo: {np.max(estados):.6f}")

    # Detectar atractor
    print("\n" + "=" * 60)
    print("DETECCIÓN DE ATRACTOR")
    print("=" * 60)
    print("\nCreando nueva red y buscando atractor...")
    red2 = RedBooleana(n_nodos=50, k=1, p_bias=0.5, semilla=123)
    periodo, transiente = red2.detectar_atractor(max_pasos=500)

    if periodo > 0:
        print(f"\n✓ Atractor encontrado!")
        print(f"  Periodo: {periodo}")
        print(f"  Transiente: {transiente}")
        if periodo == 1:
            print("  Tipo: Punto fijo")
        else:
            print(f"  Tipo: Ciclo límite de periodo {periodo}")
    else:
        print("\n✗ No se encontró atractor en 500 pasos.")

    # Visualizar historial de atractor
    if periodo > 0 and len(red2.historial) > 0:
        estados_atractor = [np.mean(estado) for estado in red2.historial]
        fig2, ax = plt.subplots(figsize=(10, 4))
        graficar_serie_temporal(np.array(estados_atractor),
                               titulo="Búsqueda de Atractor (RBN, K=1)",
                               ylabel="Estado Promedio",
                               color='purple',
                               ax=ax)
        if transiente > 0:
            ax.axvline(x=transiente, color='red', linestyle='--',
                      label=f'Inicio del atractor (t={transiente})')
            ax.legend()
        plt.savefig('examples/rbn_atractor.png', dpi=150, bbox_inches='tight')
        print("\nGráfico de atractor guardado en: examples/rbn_atractor.png")

    print("\n✓ Ejemplo completado exitosamente.")


if __name__ == "__main__":
    main()
