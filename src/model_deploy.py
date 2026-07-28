"""model_deploy.py.

Módulo responsable del despliegue (deployment) del modelo entrenado hacia
un entorno de servicio (batch o en tiempo real, según defina la
arquitectura final).

Responsabilidades de este módulo (a implementar en los siguientes avances):
    - Carga del artefacto de modelo versionado.
    - Validación de contrato de entrada/salida (esquema de features).
    - Empaquetado del modelo para servicio (API REST, batch scoring, etc.).
    - Registro del despliegue (versión de modelo, fecha, responsable) para
      trazabilidad ante auditorías.

Estado: placeholder estructural creado en V1.0.0. Lógica funcional
programada para una versión posterior (V1.3.0).

Convenciones:
    - PEP8.
    - Tipado estático (typing) en todas las firmas públicas.
    - Docstrings estilo Google en cada función pública.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def cargar_modelo(ruta_artefacto: str) -> Any:
    """Carga un artefacto de modelo serializado desde disco.

    Args:
        ruta_artefacto: Ruta al archivo del modelo serializado
            (por ejemplo, un archivo .pkl o .joblib).

    Returns:
        Objeto de modelo cargado en memoria, listo para inferencia.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.3.0)."
    )


def predecir(modelo: Any, datos_entrada: pd.DataFrame) -> pd.Series:
    """Genera predicciones sobre nuevos datos de entrada.

    Args:
        modelo: Modelo cargado en memoria.
        datos_entrada: DataFrame con las variables de entrada, ya
            transformadas según el pipeline de features.

    Returns:
        Serie de pandas con las predicciones generadas.

    Raises:
        NotImplementedError: Placeholder de estructura (V1.0.0).
    """
    raise NotImplementedError(
        "Función pendiente de implementación funcional (ver roadmap V1.3.0)."
    )


if __name__ == "__main__":
    raise SystemExit(
        "Módulo en construcción. Ejecutar únicamente vía pipeline "
        "orquestado (Jenkins) una vez implementada la lógica funcional."
    )
