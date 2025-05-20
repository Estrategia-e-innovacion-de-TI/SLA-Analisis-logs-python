<p align="center">
  <img src="docs/images/sentinel icon with text.png" alt="Sentinel Logo" width="200"/>
</p>



**Sentinel** es una herramienta simple pero poderosa para analizar logs y extraer información significativa.

Sentinel es una librería de Python diseñada para facilitar el análisis de logs de sistemas, aplicaciones y servicios. Permite a los usuarios extraer, procesar y analizar datos de logs para detectar anomalías, patrones y tendencias que podrían indicar problemas o comportamientos relevantes.  
Uno de sus principales objetivos es identificar rápidamente si los datos contienen señales que puedan indicar posibles problemas de manera proactiva.

---

## Instalación

Para instalar Sentinel, ejecuta:

```bashpdoc 
git clone https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python.git
cd SLA-Analisis-logs-python
pip install -e .
```

## Modulos

### Ingestion
Se encarga de transformar archivos de logs crudos y no estructurados en DataFrames de pandas estructurados.

Incluye un parser base y parsers específicos para:

    - WAS (WebSphere Application Server)
    - HSM (Hardware Security Module)
    - HDC (High-Density Computing)
    - IBMMQ (IBM Message Queue)

Cada parser se adapta al formato y estructura de su tipo de log respectivo.
Para tipos de logs no soportados, se pueden crear parsers personalizados basados en el parser base.

### Explorer
Ofrece herramientas para el análisis inicial de datasets enfocado en la detección de anomalías y validación de datos.

#### Funciones principales:

    - Detección de Anomalías: Detecta anomalías usando el método de rango intercuartílico (IQR).

    - Pruebas de Calidad de Datos:

        - Verifica un número mínimo de registros por columna.
        - Confirma la presencia de una columna label.
        - Evalúa el porcentaje de anomalías.
        - Valida el porcentaje mínimo de valores no nulos.
        - Evalúa umbrales mínimos de varianza.

    - Análisis de Correlación: Calcula la correlación punto-biserial entre etiquetas y características.

    - Evaluación de Modelos: Mide el recall de un modelo de regresión logística utilizando cada característica individualmente.

### Transformer
Proporciona métodos de agregación para DataFrames estructurados, útiles para datos de series de tiempo o eventos.

Herramientas disponibles:

    - StringAggregator: Agrega valores de tipo string en una ventana de tiempo definida (por ejemplo, 30 segundos).
    - RollingAggregator: Aplica agregaciones de ventanas móviles (rolling window).

### Detectors
Incluye modelos de detección de anomalías orientados a datos de series de tiempo.

Detectores disponibles:

    - AutoencoderDetector: Detección basada en autoencoders.
    - IsolationForestDetector: Detección basada en el algoritmo Isolation Forest.
    - RRCFDetector: Detección usando Robust Random Cut Forest.
    - LNNDetector: Detección usando un enfoque de Redes Neuronales Liquidas (LNN).

### Simulation
Inicializa el entorno de simulación para logs de análisis SLA y detección de anomalías en flujo (streaming).

Clase principal:

    - StreamingSimulation: Simula escenarios de detección de anomalías en flujo.
    
### Visualization
Ofrece herramientas para visualizar:

    - Resultados de detección de anomalías.
    - Análisis SHAP (SHapley Additive exPlanations) para interpretar predicciones de modelos.



# 📦 Datos del proyecto

    - Versión: 0.0.1
    - Autor: ARQUITECTURA INNOVACIÓN TI
    - Licencia: MIT