"""model_monitoring.py.

Módulo responsable del monitoreo del modelo una vez desplegado en
producción, con el fin de detectar degradación de desempeño y drift de
datos.

Responsabilidades de este módulo (a implementar en los siguientes avances):
    - Cálculo de métricas de drift (data drift, concept drift).
    - Comparación de la distribución de variables en producción vs.
      distribución de entrenamiento.
    - Registro histórico de métricas de desempeño en el tiempo.
    - Generación de alertas ante degradación significativa del modelo.

Estado: placeholder estructural creado en V1.0.0. Lógica funcional
programada para una versión posterior (V1.4.0).

Convenciones:
    - PEP8.
    - Tipado estático (typing) en todas las firmas públicas.
    - Docstrings estilo Google en cada función pública.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def calcular_drift(
    datos_referencia: pd.DataFrame,
    datos_produccion: pd.DataFrame,
) -> dict[str, float]:
    """Calcula métricas de drift entre los datos de referencia y producción.

    Args:
        datos_referencia: Datos utilizados durante el entrenamiento
            (distribución base).
        datos_produccion: Datos observados en producción a evaluar.

    Returns:
        Diccionario con métricas de drift por variable (por ejemplo,
        estadístico de Kolmogorov-Smirnov o PSI).

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.4.0)."
    )


def registrar_metricas_monitoreo(metricas: dict[str, Any]) -> None:
    """Persiste las métricas de monitoreo calculadas.

    Args:
        metricas: Diccionario de métricas a registrar (drift, desempeño,
            timestamp de ejecución, etc.).

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.4.0)."
    )


if __name__ == "__main__":
    raise SystemExit(
        "Módulo en construcción. Ejecutar únicamente vía pipeline "
        "orquestado (Jenkins) una vez implementada la lógica funcional."
    )
