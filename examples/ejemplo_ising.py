"""
Ejemplo básico del modelo de Ising.

Este script demuestra el uso básico del modelo de Ising,
incluyendo simulación y visualización del estado.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from modelos.ising import ModeloIsing
from src.visualizacion import graficar_sistema_2d, graficar_serie_temporal


def main():
    """Función principal del ejemplo."""
    print("=" * 60)
    print("EJEMPLO: MODELO DE ISING")
    print("=" * 60)

    # Parámetros
    tamanio = 50
    T_c = ModeloIsing.temperatura_critica()

    print(f"\nParámetros:")
    print(f"  Tamaño de la red: {tamanio}x{tamanio}")
    print(f"  Temperatura crítica: T_c = {T_c:.4f}")

    # Crear modelo en temperatura crítica
    print(f"\nCreando modelo de Ising en T = T_c...")
    modelo = ModeloIsing(tamanio=tamanio, temperatura=T_c, semilla=42)

    # Visualizar estado inicial
    print("\nVisualizando estado inicial...")
    fig1 = graficar_sistema_2d(modelo.obtener_estado(),
                               titulo="Estado Inicial del Modelo de Ising")
    plt.savefig('examples/ising_estado_inicial.png', dpi=150, bbox_inches='tight')
    print("  Guardado en: examples/ising_estado_inicial.png")

    # Termalización
    print("\nTermalizando el sistema (200 pasos)...")
    modelo.simular(pasos=200, calcular_magnetizacion=False)

    # Visualizar estado después de termalización
    print("\nVisualizando estado después de termalización...")
    fig2 = graficar_sistema_2d(modelo.obtener_estado(),
                               titulo="Estado tras Termalización")
    plt.savefig('examples/ising_estado_termalizado.png', dpi=150, bbox_inches='tight')
    print("  Guardado en: examples/ising_estado_termalizado.png")

    # Simular y calcular magnetización
    print("\nSimulando 1000 pasos...")
    magnetizacion = modelo.simular(pasos=1000)

    # Graficar magnetización
    print("\nGraficando evolución de la magnetización...")
    fig3, ax = plt.subplots(figsize=(10, 4))
    graficar_serie_temporal(np.abs(magnetizacion),
                           titulo="Evolución de la Magnetización (|M|)",
                           ylabel="Magnetización Absoluta",
                           color='darkblue',
                           ax=ax)
    plt.savefig('examples/ising_magnetizacion.png', dpi=150, bbox_inches='tight')
    print("  Guardado en: examples/ising_magnetizacion.png")

    # Estadísticas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS")
    print("=" * 60)
    print(f"Magnetización promedio: {np.mean(np.abs(magnetizacion)):.6f}")
    print(f"Desviación estándar: {np.std(magnetizacion):.6f}")
    print(f"Magnetización mínima: {np.min(np.abs(magnetizacion)):.6f}")
    print(f"Magnetización máxima: {np.max(np.abs(magnetizacion)):.6f}")

    print("\n✓ Ejemplo completado exitosamente.")


if __name__ == "__main__":
    main()
