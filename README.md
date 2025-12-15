# Análisis de Series Temporales en Sistemas Críticos

Este es un mensaje de prueba
Y aquí una segunda prueba

Proyecto para analizar variabilidad temporal en sistemas críticos utilizando el modelo de Ising y redes booleanas aleatorias (RBN).

## Descripción

Este proyecto implementa y compara dos modelos de sistemas críticos:

1. **Modelo de Ising**: Sistema físico de espines que presenta una transición de fase a temperatura crítica
2. **Redes Booleanas Aleatorias (RBN)**: Modelo de sistemas dinámicos discretos con transición entre regímenes ordenado, crítico y caótico

Ambos modelos son fundamentales para entender sistemas críticos en física, biología y ciencias de la complejidad.

## Características

- Simulación del modelo de Ising 2D con algoritmo de Metrópolis
- Implementación de redes booleanas aleatorias con diferentes conectividades
- Análisis de variabilidad temporal y métricas estadísticas
- Detección de comportamiento crítico
- Visualizaciones comparativas de alta calidad
- Ejemplos completos y documentados

## Estructura del Proyecto

```
analisis-series-temporales/
├── src/                          # Módulos principales
│   ├── analisis.py              # Análisis de series temporales
│   └── visualizacion.py         # Herramientas de visualización
├── modelos/                      # Implementaciones de modelos
│   ├── ising/                   # Modelo de Ising
│   │   └── modelo_ising.py
│   └── red_booleana/            # Redes booleanas
│       └── red_booleana.py
├── examples/                     # Ejemplos de uso
│   ├── comparacion_critica.py   # Comparación completa crítico vs no-crítico
│   ├── ejemplo_ising.py         # Ejemplo básico del modelo de Ising
│   └── ejemplo_rbn.py           # Ejemplo básico de RBN
├── tests/                        # Tests unitarios
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/analisis-series-temporales.git
cd analisis-series-temporales
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso Rápido

### Ejemplo 1: Comparación Completa

Ejecuta el análisis comparativo completo de ambos modelos:

```bash
python examples/comparacion_critica.py
```

Este ejemplo:
- Simula el modelo de Ising en tres regímenes (ordenado, crítico, desordenado)
- Simula redes booleanas en tres regímenes (K=1, K=2, K=3)
- Calcula variabilidad del estado promedio en 1000 pasos
- Genera gráficos comparativos y análisis estadístico

### Ejemplo 2: Modelo de Ising

```bash
python examples/ejemplo_ising.py
```

### Ejemplo 3: Redes Booleanas

```bash
python examples/ejemplo_rbn.py
```

## Uso como Biblioteca

### Modelo de Ising

```python
from modelos.ising import ModeloIsing

# Crear modelo en temperatura crítica
T_c = ModeloIsing.temperatura_critica()
modelo = ModeloIsing(tamanio=50, temperatura=T_c)

# Termalizar
modelo.simular(pasos=200, calcular_magnetizacion=False)

# Simular y obtener magnetización
magnetizacion = modelo.simular(pasos=1000)
```

### Redes Booleanas

```python
from modelos.red_booleana import RedBooleana

# Crear red en régimen crítico (K=2)
red = RedBooleana(n_nodos=100, k=2, p_bias=0.5)

# Simular
estados = red.simular(pasos=1000)

# Detectar atractor
periodo, transiente = red.detectar_atractor()
```

### Análisis y Visualización

```python
from src.analisis import calcular_metricas_completas
from src.visualizacion import graficar_variabilidad_comparativa

# Calcular métricas
metricas = calcular_metricas_completas(serie_temporal)
print(f"Variabilidad: {metricas['desviacion_estandar']:.4f}")
print(f"¿Es crítico?: {metricas['es_critico']}")

# Visualizar comparación
series = {
    'Serie 1': datos1,
    'Serie 2': datos2
}
fig = graficar_variabilidad_comparativa(series)
```

## Conceptos Teóricos

### Modelo de Ising

El modelo de Ising describe un sistema de espines que interactúan con sus vecinos. Presenta una transición de fase a la temperatura crítica:

- **T < T_c**: Régimen ordenado (magnetización espontánea)
- **T ≈ T_c**: Punto crítico (fluctuaciones a todas las escalas)
- **T > T_c**: Régimen desordenado (sin magnetización neta)

Para una red 2D: **T_c ≈ 2.269 J/k_B**

### Redes Booleanas

Las RBN muestran tres regímenes según la conectividad K:

- **K < 2**: Régimen ordenado (converge a punto fijo)
- **K = 2**: Régimen crítico (comportamiento complejo)
- **K > 2**: Régimen caótico (dinámicas impredecibles)

El punto crítico **K_c = 2** (con p_bias = 0.5) separa orden y caos.

### Criticalidad

Los sistemas críticos muestran:
- Variabilidad intermedia entre orden y caos
- Largo tiempo de correlación
- Alta entropía y complejidad
- Invarianza de escala
- Máxima capacidad de procesamiento de información

## Métricas Implementadas

- Variabilidad (desviación estándar)
- Autocorrelación temporal
- Tiempo de correlación
- Entropía de Shannon
- Detección automática de criticalidad

## Resultados Esperados

Al ejecutar `comparacion_critica.py`, se generan:

1. `comparacion_ising.png`: Dashboard del modelo de Ising
2. `comparacion_rbn.png`: Dashboard de redes booleanas
3. `comparacion_completa.png`: Comparación de todos los modelos

Los gráficos muestran:
- Series temporales
- Distribuciones de estados
- Comparación de variabilidad
- Funciones de autocorrelación

## Tests

Para ejecutar los tests:

```bash
pytest tests/
```

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Referencias

- Ising, E. (1925). "Beitrag zur Theorie des Ferromagnetismus"
- Kauffman, S. A. (1993). "The Origins of Order"
- Newman, M. & Barkema, G. (1999). "Monte Carlo Methods in Statistical Physics"
- Aldana, M. (2003). "Boolean dynamics of networks with scale-free topology"

## Licencia

MIT License

## Autor

Proyecto de análisis de sistemas críticos

## Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.
