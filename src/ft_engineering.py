"""ft_engineering.py.

Módulo de ingeniería de características (Feature Engineering) para el
pipeline de predicción de comportamiento de nuevos usuarios a partir del
histórico de créditos.

Responsabilidades de este módulo (a implementar en los siguientes avances):
    - Limpieza y transformación de variables crudas.
    - Codificación de variables categóricas.
    - Escalado / normalización de variables numéricas.
    - Creación de variables derivadas (feature engineering de negocio).
    - Serialización de los transformadores (para garantizar reproducibilidad
      entre entrenamiento e inferencia).

Estado: placeholder estructural creado en V1.0.0. Lógica funcional
programada para una versión posterior (V1.1.0), una vez cerrado el EDA.

Convenciones:
    - PEP8.
    - Tipado estático (typing) en todas las firmas públicas.
    - Docstrings estilo Google en cada función pública.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def cargar_dataframe_procesado(ruta_entrada: str) -> pd.DataFrame:
    """Carga el DataFrame intermedio generado tras la etapa de EDA.

    Args:
        ruta_entrada: Ruta al archivo intermedio (csv/parquet) generado
            después de la validación de datos.

    Returns:
        DataFrame de pandas listo para ser transformado.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0). La lógica
            se implementará en la versión de feature engineering.
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.1.0)."
    )


def construir_pipeline_features(config: dict[str, Any]) -> Any:
    """Construye el pipeline de transformación de variables.

    Args:
        config: Diccionario de configuración con los parámetros de
            transformación (columnas categóricas, numéricas, estrategia de
            imputación, etc.).

    Returns:
        Objeto pipeline (por ejemplo, sklearn.pipeline.Pipeline) listo para
        ser ajustado (fit) sobre los datos de entrenamiento.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.1.0)."
    )


if __name__ == "__main__":
    # Punto de entrada para ejecución manual / debugging local.
    raise SystemExit(
        "Módulo en construcción. Ejecutar únicamente vía pipeline "
        "orquestado (Jenkins) una vez implementada la lógica funcional."
    )
