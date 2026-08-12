"""Monitoreo de drift del pipeline de riesgo crediticio.

Compara la distribución de referencia (datos de entrenamiento,
`reference_data.csv`, generado por `model_training_evaluation.py`)
contra un lote de datos de producción, calculando:

* **Drift de variables** (feature drift): PSI, prueba KS y distancia
  de Jensen-Shannon para numéricas; PSI categórico y Chi-cuadrado para
  categóricas.
* **Drift de predicciones** (prediction drift): igual batería de
  pruebas aplicada a la probabilidad de score del modelo.
* **Drift del target** (target drift): solo cuando el desenlace real
  del crédito ya se conoce en producción. A diferencia de las
  variables predictoras, el target de un crédito (`Pago_atiempo`) no
  se observa en el momento del scoring, sino varios meses después, una
  vez el crédito madura. Por eso esta función retorna explícitamente
  `None` cuando no se provee el target real de producción, en lugar de
  forzar un cálculo con datos que en la práctica no existirían aún.

Ejecución típica en Jenkins: un job programado extrae el lote de
solicitudes scoreadas del período (día/semana) y lo compara contra la
ventana de referencia. Como este proyecto no cuenta con tráfico de
producción real, `main()` genera un lote de demostración a partir del
conjunto de prueba (`X_test`), nunca usado para entrenar el modelo,
que cumple el mismo rol que un lote de producción genuino y además
conserva el target real, permitiendo demostrar el cálculo de target
drift de punta a punta.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import (
    ConfiguracionSplit,
    VARIABLE_OBJETIVO,
    clean_data,
    load_data,
    prepare_dataset,
    split_dataset,
)
from model_training_evaluation import (
    RUTA_DATOS_REFERENCIA,
    RUTA_ESTADISTICAS_PREPROCESAMIENTO,
    RUTA_MEJOR_MODELO,
    load_model,
    predict,
)
from ft_engineering import generate_features as generar_variables_derivadas

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_monitoring")

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
RUTA_ARTEFACTOS: Final[Path] = Path("../model_artifacts")
RUTA_REPORTE_DRIFT: Final[Path] = RUTA_ARTEFACTOS / "drift_report.json"
RUTA_TABLA_DRIFT_FEATURES: Final[Path] = RUTA_ARTEFACTOS / "feature_drift.csv"

N_BINS_DRIFT: Final[int] = 10
EPSILON_PROBABILIDAD: Final[float] = 1e-4

# Umbrales estándar de la industria para interpretar el PSI (Population
# Stability Index): < 0.1 sin drift relevante, 0.1-0.25 drift moderado
# (revisar), > 0.25 drift severo (recalibrar/reentrenar).
UMBRAL_PSI_MODERADO: Final[float] = 0.10
UMBRAL_PSI_SEVERO: Final[float] = 0.25

COLUMNAS_NO_PREDICTORAS_EN_REFERENCIA: Final[list[str]] = [VARIABLE_OBJETIVO, "prediccion_proba"]


@dataclass
class ResultadoDrift:
    """Resultado estándar de una comparación de drift para una variable.

    Attributes:
        variable: Nombre de la variable evaluada.
        tipo: `"numerica"` o `"categorica"`.
        psi: Population Stability Index.
        estadistico_prueba: Estadístico de KS (numéricas) o Chi-cuadrado
            (categóricas).
        p_value: p-value de la prueba estadística correspondiente.
        distancia_js: Distancia de Jensen-Shannon (0-1).
        severidad: `"sin_drift"`, `"drift_moderado"` o `"drift_severo"`.
    """

    variable: str
    tipo: str
    psi: float
    estadistico_prueba: float
    p_value: float
    distancia_js: float
    severidad: str
    metadata_prueba: dict[str, Any] = field(default_factory=dict)


# =======================================================================
# Métricas de drift (bloques de construcción reutilizables)
# =======================================================================
def classify_drift_severity(
    psi: float, umbral_moderado: float = UMBRAL_PSI_MODERADO, umbral_severo: float = UMBRAL_PSI_SEVERO
) -> str:
    """Clasifica la severidad del drift según el valor de PSI.

    Args:
        psi: Population Stability Index calculado.
        umbral_moderado: PSI a partir del cual se considera drift moderado.
        umbral_severo: PSI a partir del cual se considera drift severo.

    Returns:
        `"sin_drift"`, `"drift_moderado"` o `"drift_severo"`.
    """
    if psi >= umbral_severo:
        return "drift_severo"
    if psi >= umbral_moderado:
        return "drift_moderado"
    return "sin_drift"


def calculate_ks_test(referencia: pd.Series, produccion: pd.Series) -> dict[str, float]:
    """Ejecuta la prueba de Kolmogorov-Smirnov de dos muestras.

    Args:
        referencia: Valores de la variable en el conjunto de referencia.
        produccion: Valores de la variable en el conjunto de producción.

    Returns:
        Diccionario con `estadistico` y `p_value`.
    """
    resultado = ks_2samp(referencia.dropna(), produccion.dropna())
    return {"estadistico": float(resultado.statistic), "p_value": float(resultado.pvalue)}


def calculate_psi(referencia: pd.Series, produccion: pd.Series, n_bins: int = N_BINS_DRIFT) -> float:
    """Calcula el Population Stability Index (PSI) para una variable numérica.

    Los puntos de corte de los `n_bins` se calculan sobre los cuantiles
    de la variable en el conjunto de REFERENCIA (nunca de producción):
    el PSI mide cuánto se aleja producción de la distribución con la
    que el modelo fue entrenado, no al revés.

    Args:
        referencia: Valores de la variable en el conjunto de referencia.
        produccion: Valores de la variable en el conjunto de producción.
        n_bins: Número de segmentos (cuantiles) a construir.

    Returns:
        PSI (>= 0). Valores más altos indican mayor desplazamiento de
        distribución.
    """
    referencia_valida = referencia.dropna()
    limites = np.unique(np.quantile(referencia_valida, np.linspace(0, 1, n_bins + 1)))
    if len(limites) < 3:
        logger.warning("PSI: variable con muy poca varianza para binning, se retorna 0.0.")
        return 0.0

    bins_referencia = pd.cut(referencia, bins=limites, include_lowest=True)
    bins_produccion = pd.cut(produccion, bins=limites, include_lowest=True)

    distribucion_referencia = bins_referencia.value_counts(normalize=True, sort=False)
    distribucion_produccion = bins_produccion.value_counts(normalize=True, sort=False).reindex(
        distribucion_referencia.index, fill_value=0
    )

    return _psi_desde_distribuciones(distribucion_referencia, distribucion_produccion)


def calculate_psi_categorical(referencia: pd.Series, produccion: pd.Series) -> float:
    """Calcula el PSI para una variable categórica (bins = categorías).

    Args:
        referencia: Valores categóricos en el conjunto de referencia.
        produccion: Valores categóricos en el conjunto de producción.

    Returns:
        PSI (>= 0).
    """
    categorias = sorted(set(referencia.dropna().unique()) | set(produccion.dropna().unique()))
    distribucion_referencia = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0)
    distribucion_produccion = produccion.value_counts(normalize=True).reindex(categorias, fill_value=0)
    return _psi_desde_distribuciones(distribucion_referencia, distribucion_produccion)


def _psi_desde_distribuciones(distribucion_referencia: pd.Series, distribucion_produccion: pd.Series) -> float:
    """Aplica la fórmula de PSI a dos distribuciones de probabilidad ya alineadas."""
    p_referencia = distribucion_referencia.clip(lower=EPSILON_PROBABILIDAD)
    p_produccion = distribucion_produccion.clip(lower=EPSILON_PROBABILIDAD)
    return float(((p_produccion - p_referencia) * np.log(p_produccion / p_referencia)).sum())


def calculate_jensen_shannon(
    referencia: pd.Series, produccion: pd.Series, es_categorica: bool = False, n_bins: int = N_BINS_DRIFT
) -> float:
    """Calcula la distancia de Jensen-Shannon entre dos distribuciones.

    Para variables numéricas, se discretiza en `n_bins` cuantiles de
    referencia (igual criterio que `calculate_psi`); para categóricas,
    se usa directamente la proporción de cada categoría.

    Args:
        referencia: Valores de la variable en el conjunto de referencia.
        produccion: Valores de la variable en el conjunto de producción.
        es_categorica: Si `True`, trata la variable como categórica.
        n_bins: Número de segmentos para variables numéricas.

    Returns:
        Distancia de Jensen-Shannon en `[0, 1]` (0 = distribuciones
        idénticas, 1 = disjuntas). Nótese que es la *distancia*
        (raíz cuadrada de la divergencia), la magnitud estándar reportada
        en monitoreo de modelos.
    """
    if es_categorica:
        categorias = sorted(set(referencia.dropna().unique()) | set(produccion.dropna().unique()))
        p = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
        q = produccion.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    else:
        referencia_valida = referencia.dropna()
        limites = np.unique(np.quantile(referencia_valida, np.linspace(0, 1, n_bins + 1)))
        if len(limites) < 3:
            logger.warning("Jensen-Shannon: variable con muy poca varianza para binning, se retorna 0.0.")
            return 0.0
        bins_referencia = pd.cut(referencia, bins=limites, include_lowest=True)
        bins_produccion = pd.cut(produccion, bins=limites, include_lowest=True)
        p = bins_referencia.value_counts(normalize=True, sort=False).to_numpy()
        q = (
            bins_produccion.value_counts(normalize=True, sort=False)
            .reindex(bins_referencia.value_counts(sort=False).index, fill_value=0)
            .to_numpy()
        )

    distancia = jensenshannon(p, q, base=2)
    return float(distancia) if np.isfinite(distancia) else 0.0


def calculate_chi_square(referencia: pd.Series, produccion: pd.Series) -> dict[str, float]:
    """Ejecuta una prueba Chi-cuadrado de homogeneidad entre dos muestras categóricas.

    Args:
        referencia: Valores categóricos en el conjunto de referencia.
        produccion: Valores categóricos en el conjunto de producción.

    Returns:
        Diccionario con `estadistico`, `p_value` y `grados_libertad`.
    """
    categorias = sorted(set(referencia.dropna().unique()) | set(produccion.dropna().unique()))
    conteos_referencia = referencia.value_counts().reindex(categorias, fill_value=0)
    conteos_produccion = produccion.value_counts().reindex(categorias, fill_value=0)
    tabla_contingencia = np.array([conteos_referencia.to_numpy(), conteos_produccion.to_numpy()])

    estadistico, p_value, grados_libertad, _ = chi2_contingency(tabla_contingencia)
    return {
        "estadistico": float(estadistico),
        "p_value": float(p_value),
        "grados_libertad": int(grados_libertad),
    }


# =======================================================================
# Orquestación de drift por tipo de variable
# =======================================================================
def _evaluar_drift_numerico(nombre: str, referencia: pd.Series, produccion: pd.Series) -> ResultadoDrift:
    """Aplica la batería completa de pruebas de drift a una variable numérica."""
    psi = calculate_psi(referencia, produccion)
    prueba_ks = calculate_ks_test(referencia, produccion)
    distancia_js = calculate_jensen_shannon(referencia, produccion, es_categorica=False)
    return ResultadoDrift(
        variable=nombre,
        tipo="numerica",
        psi=psi,
        estadistico_prueba=prueba_ks["estadistico"],
        p_value=prueba_ks["p_value"],
        distancia_js=distancia_js,
        severidad=classify_drift_severity(psi),
        metadata_prueba={"prueba": "kolmogorov_smirnov"},
    )


def _evaluar_drift_categorico(nombre: str, referencia: pd.Series, produccion: pd.Series) -> ResultadoDrift:
    """Aplica la batería completa de pruebas de drift a una variable categórica."""
    psi = calculate_psi_categorical(referencia, produccion)
    prueba_chi2 = calculate_chi_square(referencia, produccion)
    distancia_js = calculate_jensen_shannon(referencia, produccion, es_categorica=True)
    return ResultadoDrift(
        variable=nombre,
        tipo="categorica",
        psi=psi,
        estadistico_prueba=prueba_chi2["estadistico"],
        p_value=prueba_chi2["p_value"],
        distancia_js=distancia_js,
        severidad=classify_drift_severity(psi),
        metadata_prueba={"prueba": "chi_cuadrado", "grados_libertad": prueba_chi2["grados_libertad"]},
    )


def detect_feature_drift(
    df_referencia: pd.DataFrame, df_produccion: pd.DataFrame, columnas: list[str] | None = None
) -> pd.DataFrame:
    """Calcula el drift de cada variable predictora entre referencia y producción.

    El tipo de cada variable (numérica/categórica) se infiere del dtype
    en `df_referencia`, evitando mantener una lista de columnas
    duplicada y desalineada respecto a `ft_engineering.py` si el
    conjunto de variables cambia en un reentrenamiento futuro.

    Args:
        df_referencia: Variables generadas del conjunto de referencia
            (entrenamiento).
        df_produccion: Variables generadas del conjunto de producción,
            con las mismas columnas que `df_referencia`.
        columnas: Subconjunto de columnas a evaluar. Si es `None`, se
            evalúan todas las columnas comunes entre ambos DataFrames.

    Returns:
        DataFrame con una fila por variable y las columnas de
        `ResultadoDrift`, ordenado por PSI descendente.
    """
    columnas = columnas or sorted(set(df_referencia.columns) & set(df_produccion.columns))
    resultados: list[ResultadoDrift] = []

    for columna in columnas:
        try:
            es_categorica = not pd.api.types.is_numeric_dtype(df_referencia[columna])
            if es_categorica:
                resultado = _evaluar_drift_categorico(columna, df_referencia[columna], df_produccion[columna])
            else:
                resultado = _evaluar_drift_numerico(columna, df_referencia[columna], df_produccion[columna])
            resultados.append(resultado)
        except Exception:
            logger.exception("No fue posible calcular drift para la variable '%s'.", columna)

    tabla = pd.DataFrame([vars(resultado) for resultado in resultados])
    tabla = tabla.drop(columns=["metadata_prueba"]).sort_values("psi", ascending=False).reset_index(drop=True)

    n_drift_severo = int((tabla["severidad"] == "drift_severo").sum())
    n_drift_moderado = int((tabla["severidad"] == "drift_moderado").sum())
    logger.info(
        "Drift de variables: %d con drift severo, %d con drift moderado (de %d evaluadas).",
        n_drift_severo,
        n_drift_moderado,
        len(tabla),
    )
    return tabla


def detect_prediction_drift(proba_referencia: pd.Series, proba_produccion: pd.Series) -> dict[str, Any]:
    """Calcula el drift de la probabilidad de score entre referencia y producción.

    Args:
        proba_referencia: Probabilidades predichas sobre el conjunto de
            referencia (entrenamiento).
        proba_produccion: Probabilidades predichas sobre el conjunto de
            producción.

    Returns:
        Diccionario con `psi`, `ks`, `distancia_js`, `severidad` y
        estadísticas descriptivas (`media_referencia`, `media_produccion`)
        útiles para el dashboard.
    """
    psi = calculate_psi(proba_referencia, proba_produccion)
    prueba_ks = calculate_ks_test(proba_referencia, proba_produccion)
    distancia_js = calculate_jensen_shannon(proba_referencia, proba_produccion, es_categorica=False)
    resultado = {
        "psi": psi,
        "ks_estadistico": prueba_ks["estadistico"],
        "ks_p_value": prueba_ks["p_value"],
        "distancia_js": distancia_js,
        "severidad": classify_drift_severity(psi),
        "media_referencia": float(proba_referencia.mean()),
        "media_produccion": float(proba_produccion.mean()),
    }
    logger.info("Drift de predicciones: PSI=%.4f (%s)", psi, resultado["severidad"])
    return resultado


def detect_target_drift(
    y_referencia: pd.Series, y_produccion: pd.Series | None
) -> dict[str, Any] | None:
    """Calcula el drift del target real, cuando el desenlace ya se conoce.

    A diferencia de las variables predictoras y de la probabilidad de
    score (disponibles al momento del scoring), el target real de un
    crédito solo se conoce meses después de originado, cuando el
    crédito madura. Por eso esta función retorna `None` de forma
    explícita si `y_produccion` no está disponible, en lugar de
    fabricar un resultado con datos que en producción real no
    existirían aún.

    Args:
        y_referencia: Target real del conjunto de referencia (entrenamiento).
        y_produccion: Target real observado en producción, o `None` si
            aún no se conoce.

    Returns:
        Diccionario con `psi`, prueba `chi_cuadrado`, `tasa_positivos_referencia`
        y `tasa_positivos_produccion`; o `None` si `y_produccion` es `None`.
    """
    if y_produccion is None:
        logger.info("Target drift no calculado: aún no se dispone del desenlace real en producción.")
        return None

    y_referencia_str = y_referencia.astype(str)
    y_produccion_str = y_produccion.astype(str)
    psi = calculate_psi_categorical(y_referencia_str, y_produccion_str)
    prueba_chi2 = calculate_chi_square(y_referencia_str, y_produccion_str)
    resultado = {
        "psi": psi,
        "chi_cuadrado_estadistico": prueba_chi2["estadistico"],
        "chi_cuadrado_p_value": prueba_chi2["p_value"],
        "severidad": classify_drift_severity(psi),
        "tasa_positivos_referencia": float(y_referencia.mean()),
        "tasa_positivos_produccion": float(y_produccion.mean()),
    }
    logger.info("Drift de target: PSI=%.4f (%s)", psi, resultado["severidad"])
    return resultado


# =======================================================================
# Carga de datos
# =======================================================================
def load_reference_data(ruta: Path = RUTA_DATOS_REFERENCIA) -> pd.DataFrame:
    """Carga la instantánea de referencia generada en entrenamiento.

    Args:
        ruta: Ruta al archivo `reference_data.csv`.

    Returns:
        DataFrame con las variables generadas, el target real
        (`VARIABLE_OBJETIVO`) y `prediccion_proba`.

    Raises:
        FileNotFoundError: Si el archivo no existe (el pipeline de
            entrenamiento debe ejecutarse antes que el de monitoreo).
    """
    if not ruta.exists():
        mensaje = (
            f"No se encontraron datos de referencia en {ruta}. "
            "Ejecute model_training_evaluation.py antes de model_monitoring.py."
        )
        logger.error(mensaje)
        raise FileNotFoundError(mensaje)
    return pd.read_csv(ruta)


def cargar_estadisticas_preprocesamiento(ruta: Path = RUTA_ESTADISTICAS_PREPROCESAMIENTO) -> dict[str, Any]:
    """Carga las estadísticas de referencia usadas para limpieza/ingeniería de variables.

    Misma lógica que `model_deploy.cargar_estadisticas_preprocesamiento`:
    se reutiliza aquí en vez de importarla directamente porque
    `model_deploy.py` no expone la función como parte de su contrato
    público de reuso (sí lo hace vía sus propios endpoints), evitando un
    acoplamiento circular entre el servicio de inferencia y el de
    monitoreo.

    Args:
        ruta: Ruta al archivo `preprocessing_stats.json`.

    Returns:
        Diccionario con `medianas_referencia` y `limites_winsorizacion`.
    """
    with ruta.open("r", encoding="utf-8") as archivo:
        estadisticas = json.load(archivo)
    estadisticas["limites_winsorizacion"] = {
        columna: tuple(limites) for columna, limites in estadisticas["limites_winsorizacion"].items()
    }
    return estadisticas


def generar_datos_produccion_ejemplo() -> tuple[pd.DataFrame, pd.Series]:
    """Genera un lote de demostración que emula un extracto de producción.

    Reconstruye exactamente el mismo split de entrenamiento (misma
    `RANDOM_STATE`, mismas funciones de `ft_engineering`) para recuperar
    las filas CRUDAS correspondientes a `X_test`: un conjunto nunca visto
    durante el entrenamiento, por lo que cumple honestamente el rol de
    "producción" para esta demostración. Al provenir del dataset
    histórico, conserva el target real, lo que permite demostrar también
    el cálculo de target drift (ver docstring del módulo).

    En un entorno productivo real, esta función se reemplaza por la
    lectura del extracto de solicitudes efectivamente scoreadas por
    `model_deploy.py` en el período de monitoreo.

    Returns:
        Tupla `(dataframe_crudo_produccion, target_real_produccion)`.
    """
    dataframe_crudo = load_data()
    X, y, _, _, _ = prepare_dataset(dataframe_crudo)
    _, X_test, _, y_test = split_dataset(X, y, ConfiguracionSplit(proporcion_test=0.2))

    df_produccion_cruda = dataframe_crudo.loc[X_test.index].drop(columns=[VARIABLE_OBJETIVO])
    logger.info("Lote de producción de ejemplo generado a partir de X_test: %d filas.", len(df_produccion_cruda))
    return df_produccion_cruda, y_test


def preparar_produccion_para_scoring(
    dataframe_crudo_produccion: pd.DataFrame, estadisticas_preprocesamiento: dict[str, Any]
) -> pd.DataFrame:
    """Aplica la misma limpieza/ingeniería de variables usada en `model_deploy.py`.

    Args:
        dataframe_crudo_produccion: Lote crudo de solicitudes de producción.
        estadisticas_preprocesamiento: Medianas y límites de winsorización
            de referencia (entrenamiento).

    Returns:
        DataFrame con las variables generadas, listo para `predict()`.
    """
    df_limpio, _ = clean_data(
        dataframe_crudo_produccion, medianas_referencia=estadisticas_preprocesamiento["medianas_referencia"]
    )
    df_features, _ = generar_variables_derivadas(
        df_limpio, limites_winsorizacion=estadisticas_preprocesamiento["limites_winsorizacion"]
    )
    return df_features


# =======================================================================
# Reporte consolidado
# =======================================================================
def generate_monitoring_report(
    df_referencia: pd.DataFrame,
    df_produccion: pd.DataFrame,
    proba_referencia: pd.Series,
    proba_produccion: pd.Series,
    y_referencia: pd.Series,
    y_produccion: pd.Series | None,
) -> dict[str, Any]:
    """Genera el reporte de monitoreo completo (features + predicción + target).

    Args:
        df_referencia: Variables generadas del conjunto de referencia.
        df_produccion: Variables generadas del conjunto de producción.
        proba_referencia: Probabilidades predichas sobre referencia.
        proba_produccion: Probabilidades predichas sobre producción.
        y_referencia: Target real de referencia.
        y_produccion: Target real de producción (`None` si aún no se conoce).

    Returns:
        Diccionario serializable con el resumen ejecutivo y el detalle
        de drift por variable.
    """
    columnas_comunes = sorted(
        set(df_referencia.columns) & set(df_produccion.columns) - set(COLUMNAS_NO_PREDICTORAS_EN_REFERENCIA)
    )
    tabla_drift_features = detect_feature_drift(df_referencia, df_produccion, columnas=columnas_comunes)
    resumen_prediction_drift = detect_prediction_drift(proba_referencia, proba_produccion)
    resumen_target_drift = detect_target_drift(y_referencia, y_produccion)

    reporte = {
        "n_referencia": len(df_referencia),
        "n_produccion": len(df_produccion),
        "resumen_ejecutivo": {
            "variables_con_drift_severo": tabla_drift_features.loc[
                tabla_drift_features["severidad"] == "drift_severo", "variable"
            ].tolist(),
            "variables_con_drift_moderado": tabla_drift_features.loc[
                tabla_drift_features["severidad"] == "drift_moderado", "variable"
            ].tolist(),
            "drift_prediccion": resumen_prediction_drift["severidad"],
            "drift_target": resumen_target_drift["severidad"] if resumen_target_drift else "no_disponible",
        },
        "drift_predicciones": resumen_prediction_drift,
        "drift_target": resumen_target_drift,
        "drift_variables": tabla_drift_features.to_dict(orient="records"),
    }
    return reporte


# =======================================================================
# main
# =======================================================================
def main() -> None:
    """Orquesta la comparación de drift entre entrenamiento y producción."""
    try:
        RUTA_ARTEFACTOS.mkdir(parents=True, exist_ok=True)

        df_referencia_completa = load_reference_data()
        y_referencia = df_referencia_completa[VARIABLE_OBJETIVO]
        proba_referencia = df_referencia_completa["prediccion_proba"]
        df_referencia = df_referencia_completa.drop(columns=COLUMNAS_NO_PREDICTORAS_EN_REFERENCIA)

        pipeline = load_model(RUTA_MEJOR_MODELO)
        estadisticas_preprocesamiento = cargar_estadisticas_preprocesamiento()

        dataframe_crudo_produccion, y_produccion = generar_datos_produccion_ejemplo()
        df_produccion = preparar_produccion_para_scoring(dataframe_crudo_produccion, estadisticas_preprocesamiento)
        _, proba_produccion_array = predict(pipeline, df_produccion)
        proba_produccion = pd.Series(proba_produccion_array, index=df_produccion.index)

        reporte = generate_monitoring_report(
            df_referencia, df_produccion, proba_referencia, proba_produccion, y_referencia, y_produccion
        )

        with RUTA_REPORTE_DRIFT.open("w", encoding="utf-8") as archivo_reporte:
            json.dump(reporte, archivo_reporte, indent=2, ensure_ascii=False)
        pd.DataFrame(reporte["drift_variables"]).to_csv(RUTA_TABLA_DRIFT_FEATURES, index=False)

        logger.info("Reporte de monitoreo guardado en %s", RUTA_REPORTE_DRIFT)
        logger.info("Tabla de drift de variables guardada en %s", RUTA_TABLA_DRIFT_FEATURES)
        logger.info("Resumen ejecutivo: %s", reporte["resumen_ejecutivo"])

    except Exception:
        logger.exception("Fallo no controlado en model_monitoring.main()")
        raise


if __name__ == "__main__":
    main()
