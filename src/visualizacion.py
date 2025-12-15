"""
Módulo de visualización de series temporales y análisis.

Este módulo proporciona herramientas para visualizar series temporales
y sus propiedades estadísticas.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns


# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")


def graficar_serie_temporal(serie_temporal: np.ndarray,
                           titulo: str = "Serie Temporal",
                           xlabel: str = "Tiempo",
                           ylabel: str = "Valor",
                           color: str = "blue",
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Grafica una serie temporal.

    Args:
        serie_temporal: Array 1D con los valores de la serie temporal
        titulo: Título del gráfico
        xlabel: Etiqueta del eje x
        ylabel: Etiqueta del eje y
        color: Color de la línea
        ax: Axes de matplotlib (opcional)

    Returns:
        Axes de matplotlib con el gráfico
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(serie_temporal, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)

    return ax


def graficar_comparacion(series: Dict[str, np.ndarray],
                        titulo: str = "Comparación de Series Temporales",
                        xlabel: str = "Tiempo",
                        ylabel: str = "Valor",
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compara múltiples series temporales en el mismo gráfico.

    Args:
        series: Diccionario con nombre y serie temporal
        titulo: Título del gráfico
        xlabel: Etiqueta del eje x
        ylabel: Etiqueta del eje y
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    colores = plt.cm.tab10(np.linspace(0, 1, len(series)))

    for (nombre, serie), color in zip(series.items(), colores):
        ax.plot(serie, label=nombre, linewidth=1.5, alpha=0.8, color=color)

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def graficar_variabilidad_comparativa(series: Dict[str, np.ndarray],
                                     figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Crea un dashboard comparativo de variabilidad entre diferentes series.

    Args:
        series: Diccionario con nombre y serie temporal
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib con múltiples subplots
    """
    from src.analisis import calcular_variabilidad, calcular_autocorrelacion

    n_series = len(series)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Series temporales
    ax1 = axes[0, 0]
    colores = plt.cm.tab10(np.linspace(0, 1, n_series))
    for (nombre, serie), color in zip(series.items(), colores):
        ax1.plot(serie, label=nombre, linewidth=1.5, alpha=0.7, color=color)
    ax1.set_title("Series Temporales", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Tiempo")
    ax1.set_ylabel("Estado Promedio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogramas de distribución
    ax2 = axes[0, 1]
    for (nombre, serie), color in zip(series.items(), colores):
        ax2.hist(serie, bins=30, alpha=0.5, label=nombre, color=color, density=True)
    ax2.set_title("Distribución de Estados", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Densidad")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Comparación de variabilidad
    ax3 = axes[1, 0]
    nombres = list(series.keys())
    variabilidades = [calcular_variabilidad(serie) for serie in series.values()]
    bars = ax3.bar(nombres, variabilidades, color=colores, alpha=0.7)
    ax3.set_title("Variabilidad (Desviación Estándar)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Desviación Estándar")
    ax3.grid(True, alpha=0.3, axis='y')

    # Añadir valores sobre las barras
    for bar, val in zip(bars, variabilidades):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. Autocorrelación
    ax4 = axes[1, 1]
    for (nombre, serie), color in zip(series.items(), colores):
        autocorr = calcular_autocorrelacion(serie, max_lag=100)
        ax4.plot(autocorr, label=nombre, linewidth=2, alpha=0.7, color=color)
    ax4.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.5, label='1/e')
    ax4.set_title("Función de Autocorrelación", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelación")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def graficar_sistema_2d(estado: np.ndarray,
                       titulo: str = "Estado del Sistema",
                       cmap: str = "RdBu",
                       figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Visualiza el estado de un sistema 2D (como el modelo de Ising).

    Args:
        estado: Matriz 2D con el estado del sistema
        titulo: Título del gráfico
        cmap: Mapa de colores
        figsize: Tamaño de la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(estado, cmap=cmap, interpolation='nearest')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Valor')

    plt.tight_layout()
    return fig


def guardar_graficos(fig: plt.Figure,
                    nombre_archivo: str,
                    dpi: int = 300,
                    formato: str = 'png'):
    """
    Guarda una figura en un archivo.

    Args:
        fig: Figura de matplotlib
        nombre_archivo: Nombre del archivo (sin extensión)
        dpi: Resolución en puntos por pulgada
        formato: Formato del archivo (png, pdf, svg, etc.)
    """
    fig.savefig(f"{nombre_archivo}.{formato}", dpi=dpi, bbox_inches='tight')
    print(f"Gráfico guardado en: {nombre_archivo}.{formato}")
