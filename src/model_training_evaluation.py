"""model_training_evaluation.py.

Módulo responsable del entrenamiento y evaluación del modelo de Machine
Learning para la predicción del comportamiento de nuevos usuarios en base
al histórico de créditos.

Responsabilidades de este módulo (a implementar en los siguientes avances):
    - Split de datos (train / test / validación) con control de semilla
      para reproducibilidad.
    - Entrenamiento del/los modelo(s) candidato(s).
    - Validación cruzada y búsqueda de hiperparámetros.
    - Cálculo de métricas de negocio y estadísticas (AUC, KS, precision,
      recall, F1, entre otras relevantes para riesgo crediticio).
    - Registro de experimentos (tracking) para trazabilidad.
    - Serialización del modelo entrenado (artefacto versionado).

Estado: placeholder estructural creado en V1.0.0. Lógica funcional
programada para una versión posterior (V1.2.0).

Convenciones:
    - PEP8.
    - Tipado estático (typing) en todas las firmas públicas.
    - Docstrings estilo Google en cada función pública.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def entrenar_modelo(
    x_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    parametros: dict[str, Any] | None = None,
) -> Any:
    """Entrena el modelo de clasificación sobre el conjunto de entrenamiento.

    Args:
        x_entrenamiento: Variables predictoras de entrenamiento.
        y_entrenamiento: Variable objetivo de entrenamiento.
        parametros: Hiperparámetros del modelo. Si es ``None`` se utilizan
            los valores por defecto definidos en configuración.

    Returns:
        Objeto de modelo entrenado.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.2.0)."
    )


def evaluar_modelo(
    modelo: Any,
    x_prueba: pd.DataFrame,
    y_prueba: pd.Series,
) -> dict[str, float]:
    """Evalúa el modelo entrenado sobre el conjunto de prueba.

    Args:
        modelo: Modelo previamente entrenado.
        x_prueba: Variables predictoras de prueba.
        y_prueba: Variable objetivo de prueba.

    Returns:
        Diccionario con las métricas de evaluación calculadas.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.2.0)."
    )


if __name__ == "__main__":
    raise SystemExit(
        "Módulo en construcción. Ejecutar únicamente vía pipeline "
        "orquestado (Jenkins) una vez implementada la lógica funcional."
    )
