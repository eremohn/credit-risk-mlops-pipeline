"""Feature engineering para el pipeline de riesgo crediticio.

Este módulo transforma la salida validada de ``cargar_datos.ipynb`` en
matrices de entrenamiento/prueba listas para modelamiento, aplicando
exactamente las decisiones documentadas en ``comprension_eda.ipynb``:
exclusión de variables con fuga de información temporal (saldos
post-originación), corrección de inconsistencias, generación de
variables derivadas y un pipeline de preprocesamiento reutilizable
(``ColumnTransformer``) que se serializa junto con el modelo entrenado
para garantizar paridad entrenamiento/inferencia.

No debe ejecutarse EDA exploratorio aquí: esa responsabilidad ya fue
cubierta por los notebooks existentes. Este módulo solo *aplica* esas
decisiones de forma determinística y reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ft_engineering")

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42

RUTA_DATOS_DEFECTO: Final[Path] = Path("../base_de_datos.csv")
VARIABLE_OBJETIVO: Final[str] = "Pago_atiempo"

# Columnas con fuga de información: describen el estado del crédito
# DESPUÉS de originado. Nunca deben usarse como predictoras (ver
# comprension_eda.ipynb, sección 10.1).
COLUMNAS_FUGA_INFORMACION: Final[list[str]] = [
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
]

# Columnas monetarias con sesgo alto: candidatas a log1p (sección 10.3).
COLUMNAS_LOG: Final[list[str]] = [
    "capital_prestado",
    "salario_cliente",
    "total_otros_prestamos",
    "promedio_ingresos_datacredito",
]

# Columnas a winsorizar antes del log (decisión de negocio, sección 6).
COLUMNAS_WINSORIZAR: Final[list[str]] = [
    "salario_cliente",
    "total_otros_prestamos",
    "promedio_ingresos_datacredito",
]
PERCENTIL_WINSORIZAR_INFERIOR: Final[float] = 0.01
PERCENTIL_WINSORIZAR_SUPERIOR: Final[float] = 0.99

EDAD_MAXIMA_PLAUSIBLE: Final[int] = 100
CATEGORIAS_VALIDAS_TENDENCIA: Final[set[str]] = {
    "Estable",
    "Creciente",
    "Decreciente",
}

# Variables finales del modelo (post feature-engineering).
COLUMNAS_NUMERICAS_MODELO: Final[list[str]] = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "ratio_cuota_salario",
    "mes_prestamo",
    "trimestre_prestamo",
    "total_creditos_sector",
    "edad_atipica",
    "sin_promedio_ingresos",
]
COLUMNAS_CATEGORICAS_MODELO: Final[list[str]] = [
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos",
    "categoria_riesgo_score",
]

UMBRAL_CORRELACION_REDUNDANTE: Final[float] = 0.9


@dataclass(frozen=True)
class ConfiguracionSplit:
    """Parámetros de la división train/test.

    Attributes:
        proporcion_test: Fracción de los datos reservada para prueba.
        estratificar: Si ``True``, estratifica por `VARIABLE_OBJETIVO`.
        random_state: Semilla de aleatoriedad para reproducibilidad.
    """

    proporcion_test: float = 0.2
    estratificar: bool = True
    random_state: int = field(default=RANDOM_STATE)


# ---------------------------------------------------------------------
# Funciones: carga
# ---------------------------------------------------------------------
def load_data(ruta: Path | str = RUTA_DATOS_DEFECTO) -> pd.DataFrame:
    """Carga la fuente histórica de créditos desde disco.

    Reutiliza el mismo contrato de datos validado en
    ``cargar_datos.ipynb`` (23 columnas, `fecha_prestamo` como fecha).
    No repite las validaciones de esquema del notebook: asume que la
    fuente ya fue auditada aguas arriba en el pipeline de Jenkins.

    Args:
        ruta: Ruta al archivo CSV de origen.

    Returns:
        DataFrame crudo con `fecha_prestamo` parseada como fecha.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o no puede parsearse.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        mensaje = f"No se encontró la fuente de datos en: {ruta}"
        logger.error(mensaje)
        raise FileNotFoundError(mensaje)

    try:
        dataframe = pd.read_csv(ruta, encoding="utf-8", parse_dates=["fecha_prestamo"])
    except pd.errors.EmptyDataError as error:
        mensaje = f"El archivo {ruta} está vacío."
        logger.error(mensaje)
        raise ValueError(mensaje) from error
    except pd.errors.ParserError as error:
        mensaje = f"Error de parseo al leer {ruta}: {error}"
        logger.error(mensaje)
        raise ValueError(mensaje) from error

    logger.info("Datos cargados: %d filas x %d columnas", *dataframe.shape)
    return dataframe


# ---------------------------------------------------------------------
# Funciones: limpieza (réplica determinística de comprension_eda.ipynb)
# ---------------------------------------------------------------------
def _corregir_puntajes_invalidos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convierte puntajes negativos en NaN para imputación posterior."""
    dataframe = dataframe.copy()
    for columna in ("puntaje", "puntaje_datacredito"):
        n_invalidos = (dataframe[columna] < 0).sum()
        if n_invalidos:
            logger.warning("%s: %d valores negativos invalidados", columna, n_invalidos)
        dataframe.loc[dataframe[columna] < 0, columna] = np.nan
    return dataframe


def _corregir_tendencia_ingresos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Invalida categorías espurias fuera del dominio de negocio."""
    dataframe = dataframe.copy()
    es_invalida = ~dataframe["tendencia_ingresos"].isin(
        CATEGORIAS_VALIDAS_TENDENCIA
    ) & dataframe["tendencia_ingresos"].notna()
    if es_invalida.any():
        logger.warning(
            "tendencia_ingresos: %d valores fuera de dominio invalidados",
            int(es_invalida.sum()),
        )
    dataframe.loc[es_invalida, "tendencia_ingresos"] = np.nan
    dataframe["tendencia_ingresos"] = dataframe["tendencia_ingresos"].fillna("Sin_dato")
    return dataframe


def _corregir_edad(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Capa edades imposibles y deja una bandera de auditoría."""
    dataframe = dataframe.copy()
    dataframe["edad_atipica"] = (dataframe["edad_cliente"] > EDAD_MAXIMA_PLAUSIBLE).astype(int)
    dataframe.loc[dataframe["edad_cliente"] > EDAD_MAXIMA_PLAUSIBLE, "edad_cliente"] = (
        EDAD_MAXIMA_PLAUSIBLE
    )
    return dataframe


def calculate_winsorize_bounds(
    dataframe: pd.DataFrame,
    columnas: list[str] = COLUMNAS_WINSORIZAR,
    percentil_inferior: float = PERCENTIL_WINSORIZAR_INFERIOR,
    percentil_superior: float = PERCENTIL_WINSORIZAR_SUPERIOR,
) -> dict[str, tuple[float, float]]:
    """Calcula los límites de winsorización de referencia (p.ej. de entrenamiento).

    Estos límites deben calcularse una única vez sobre el dataset de
    entrenamiento y reutilizarse tal cual en inferencia. Recalcularlos
    sobre cada lote de solicitudes entrantes produciría *train/serve
    skew*: un lote pequeño (o de una sola solicitud) generaría límites
    de recorte arbitrarios en lugar de reflejar la distribución con la
    que el modelo fue entrenado.

    Args:
        dataframe: Dataset de referencia (normalmente el de entrenamiento).
        columnas: Columnas numéricas a winsorizar.
        percentil_inferior: Percentil inferior de corte (0-1).
        percentil_superior: Percentil superior de corte (0-1).

    Returns:
        Diccionario `{columna: (limite_inferior, limite_superior)}`.
    """
    return {
        columna: (
            float(dataframe[columna].quantile(percentil_inferior)),
            float(dataframe[columna].quantile(percentil_superior)),
        )
        for columna in columnas
    }


def winsorize_column(
    serie: pd.Series,
    percentil_inferior: float = PERCENTIL_WINSORIZAR_INFERIOR,
    percentil_superior: float = PERCENTIL_WINSORIZAR_SUPERIOR,
) -> pd.Series:
    """Recorta una serie numérica a sus propios percentiles p1-p99.

    Uso exclusivo en contexto de entrenamiento (recalcula los límites a
    partir de la propia serie). Para aplicar límites ya calculados sobre
    un dataset de referencia — el caso correcto en inferencia — usar
    `apply_winsorize_bounds`.

    Args:
        serie: Serie numérica a winsorizar.
        percentil_inferior: Percentil inferior de corte (0-1).
        percentil_superior: Percentil superior de corte (0-1).

    Returns:
        Serie con valores extremos recortados a los límites calculados.
    """
    limite_inferior = serie.quantile(percentil_inferior)
    limite_superior = serie.quantile(percentil_superior)
    return serie.clip(lower=limite_inferior, upper=limite_superior)


def apply_winsorize_bounds(serie: pd.Series, limites: tuple[float, float]) -> pd.Series:
    """Recorta una serie a límites ya calculados (uso en inferencia).

    Args:
        serie: Serie numérica a recortar.
        limites: Tupla `(limite_inferior, limite_superior)` obtenida con
            `calculate_winsorize_bounds` sobre el dataset de entrenamiento.

    Returns:
        Serie recortada a los límites dados.
    """
    limite_inferior, limite_superior = limites
    return serie.clip(lower=limite_inferior, upper=limite_superior)


def clean_data(
    dataframe: pd.DataFrame, medianas_referencia: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aplica la limpieza determinística documentada en el EDA.

    Corrige inconsistencias (puntajes negativos, categorías espurias,
    edades imposibles) e imputa nulos de forma explícita, dejando el
    DataFrame listo para la generación de variables derivadas.

    Args:
        dataframe: DataFrame crudo tal como retorna `load_data`.
        medianas_referencia: Medianas ya calculadas sobre el dataset de
            entrenamiento, en formato `{columna: mediana}`. Si se
            proveen, se usan directamente para imputar (caso de
            inferencia, evita *train/serve skew*). Si es ``None``, las
            medianas se calculan a partir del propio `dataframe` (caso
            de entrenamiento, comportamiento histórico de esta función).

    Returns:
        Tupla `(dataframe_limpio, medianas_utilizadas)`. `medianas_utilizadas`
        debe persistirse en entrenamiento y reutilizarse en inferencia.
    """
    try:
        df_limpio = dataframe.copy()
        df_limpio = _corregir_puntajes_invalidos(df_limpio)
        df_limpio = _corregir_tendencia_ingresos(df_limpio)
        df_limpio = _corregir_edad(df_limpio)

        df_limpio["sin_promedio_ingresos"] = (
            df_limpio["promedio_ingresos_datacredito"].isna().astype(int)
        )

        columnas_a_imputar = ("puntaje", "puntaje_datacredito", "promedio_ingresos_datacredito")
        medianas_utilizadas = dict(medianas_referencia) if medianas_referencia else {}
        for columna in columnas_a_imputar:
            if columna not in medianas_utilizadas:
                medianas_utilizadas[columna] = float(df_limpio[columna].median())
            df_limpio[columna] = df_limpio[columna].fillna(medianas_utilizadas[columna])

        columnas_saldo_presentes = [
            columna for columna in COLUMNAS_FUGA_INFORMACION if columna in df_limpio.columns
        ]
        df_limpio[columnas_saldo_presentes] = df_limpio[columnas_saldo_presentes].fillna(0)

    except KeyError as error:
        mensaje = f"Columna esperada ausente durante la limpieza: {error}"
        logger.error(mensaje)
        raise ValueError(mensaje) from error

    logger.info("Limpieza aplicada. Nulos remanentes: %d", int(df_limpio.isna().sum().sum()))
    return df_limpio, medianas_utilizadas


# ---------------------------------------------------------------------
# Funciones: ingeniería de variables
# ---------------------------------------------------------------------
def generate_features(
    dataframe: pd.DataFrame, limites_winsorizacion: dict[str, tuple[float, float]] | None = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Genera las variables derivadas prototipadas en el EDA.

    Crea razones financieras, componentes temporales de la fecha de
    desembolso, agregados de sector y un bucket de riesgo por score
    externo. Además aplica winsorización + transformación logarítmica
    a las variables monetarias sesgadas y elimina, de forma explícita,
    las columnas con fuga de información temporal.

    Args:
        dataframe: DataFrame limpio, salida de `clean_data`.
        limites_winsorizacion: Límites ya calculados sobre el dataset de
            entrenamiento, en formato `{columna: (p1, p99)}`. Si se
            proveen, se aplican directamente (caso de inferencia, evita
            *train/serve skew*). Si es ``None``, los límites se calculan
            a partir del propio `dataframe` (caso de entrenamiento).

    Returns:
        Tupla `(dataframe_enriquecido, limites_utilizados)`. `dataframe_enriquecido`
        no contiene columnas de fuga de información ni la fecha cruda.
        `limites_utilizados` debe persistirse en entrenamiento y
        reutilizarse en inferencia.
    """
    try:
        df_feat = dataframe.copy()

        df_feat["ratio_cuota_salario"] = (
            df_feat["cuota_pactada"] / df_feat["salario_cliente"].replace(0, np.nan)
        ).fillna(0.0)
        df_feat["mes_prestamo"] = df_feat["fecha_prestamo"].dt.month
        df_feat["trimestre_prestamo"] = df_feat["fecha_prestamo"].dt.quarter
        df_feat["total_creditos_sector"] = (
            df_feat["creditos_sectorFinanciero"]
            + df_feat["creditos_sectorCooperativo"]
            + df_feat["creditos_sectorReal"]
        )
        df_feat["categoria_riesgo_score"] = pd.cut(
            df_feat["puntaje_datacredito"],
            bins=[0, 500, 700, 850, 999],
            labels=["alto_riesgo", "riesgo_medio", "riesgo_bajo", "riesgo_muy_bajo"],
        ).astype(object).fillna("sin_clasificar")

        limites_utilizados = dict(limites_winsorizacion) if limites_winsorizacion else (
            calculate_winsorize_bounds(df_feat, COLUMNAS_WINSORIZAR)
        )
        for columna in COLUMNAS_WINSORIZAR:
            df_feat[columna] = apply_winsorize_bounds(df_feat[columna], limites_utilizados[columna])
        for columna in COLUMNAS_LOG:
            df_feat[columna] = np.log1p(df_feat[columna].clip(lower=0))

        columnas_a_eliminar = [
            columna
            for columna in (*COLUMNAS_FUGA_INFORMACION, "fecha_prestamo")
            if columna in df_feat.columns
        ]
        df_feat = df_feat.drop(columns=columnas_a_eliminar)

    except KeyError as error:
        mensaje = f"Columna esperada ausente al generar variables: {error}"
        logger.error(mensaje)
        raise ValueError(mensaje) from error

    logger.info(
        "Variables derivadas generadas. Shape resultante: %d x %d", *df_feat.shape
    )
    return df_feat, limites_utilizados


def select_features(
    dataframe: pd.DataFrame,
    columnas_numericas: list[str],
    umbral_correlacion: float = UMBRAL_CORRELACION_REDUNDANTE,
) -> list[str]:
    """Descarta variables numéricas redundantes por alta correlación.

    Cuando dos variables tienen correlación absoluta por encima del
    umbral, se conserva la que tiene mayor correlación con el target y
    se descarta la otra, evitando eliminar señal de forma arbitraria.

    Args:
        dataframe: DataFrame que contiene las columnas a evaluar y el
            target (`VARIABLE_OBJETIVO`).
        columnas_numericas: Lista de columnas numéricas candidatas.
        umbral_correlacion: Umbral de correlación absoluta a partir del
            cual dos variables se consideran redundantes.

    Returns:
        Lista de columnas numéricas depurada, sin pares redundantes.
    """
    matriz_corr = dataframe[columnas_numericas].corr().abs()
    corr_target = dataframe[columnas_numericas].corrwith(dataframe[VARIABLE_OBJETIVO]).abs()

    columnas_descartadas: set[str] = set()
    for i, columna_a in enumerate(columnas_numericas):
        for columna_b in columnas_numericas[i + 1 :]:
            if columna_a in columnas_descartadas or columna_b in columnas_descartadas:
                continue
            if matriz_corr.loc[columna_a, columna_b] > umbral_correlacion:
                peor = columna_b if corr_target[columna_a] >= corr_target[columna_b] else columna_a
                logger.warning(
                    "Variable redundante descartada: %s (correlación con %s = %.3f)",
                    peor,
                    columna_a if peor == columna_b else columna_b,
                    matriz_corr.loc[columna_a, columna_b],
                )
                columnas_descartadas.add(peor)

    return [columna for columna in columnas_numericas if columna not in columnas_descartadas]


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict[str, Any]]:
    """Orquesta limpieza, generación y selección de variables.

    Args:
        dataframe: DataFrame crudo, salida de `load_data`.

    Returns:
        Tupla `(X, y, columnas_numericas, columnas_categoricas,
        estadisticas_preprocesamiento)` lista para dividir en train/test
        y alimentar el pipeline de preprocesamiento.
        `estadisticas_preprocesamiento` contiene `medianas_referencia` y
        `limites_winsorizacion`: deben persistirse junto con el modelo y
        reutilizarse tal cual en `model_deploy.py` para evitar
        *train/serve skew* (ver docstrings de `clean_data` y
        `generate_features`).
    """
    df_limpio, medianas_referencia = clean_data(dataframe)
    df_features, limites_winsorizacion = generate_features(df_limpio)

    columnas_numericas = select_features(df_features, COLUMNAS_NUMERICAS_MODELO)
    columnas_categoricas = COLUMNAS_CATEGORICAS_MODELO

    columnas_predictoras = columnas_numericas + columnas_categoricas
    X = df_features[columnas_predictoras]
    y = df_features[VARIABLE_OBJETIVO]

    estadisticas_preprocesamiento = {
        "medianas_referencia": medianas_referencia,
        "limites_winsorizacion": limites_winsorizacion,
    }

    logger.info(
        "Dataset preparado: %d predictoras (%d numéricas, %d categóricas)",
        len(columnas_predictoras),
        len(columnas_numericas),
        len(columnas_categoricas),
    )
    return X, y, columnas_numericas, columnas_categoricas, estadisticas_preprocesamiento


# ---------------------------------------------------------------------
# Funciones: preprocesamiento y split
# ---------------------------------------------------------------------
def create_preprocessing_pipeline(
    columnas_numericas: list[str], columnas_categoricas: list[str]
) -> ColumnTransformer:
    """Construye el `ColumnTransformer` de preprocesamiento del modelo.

    Numéricas: imputación por mediana + escalado estándar (necesario
    para Regresión Logística; neutro para modelos basados en árboles).
    Categóricas: imputación por moda + one-hot encoding, ignorando
    categorías no vistas en inferencia para no romper el servicio.

    Args:
        columnas_numericas: Nombres de las columnas numéricas.
        columnas_categoricas: Nombres de las columnas categóricas.

    Returns:
        `ColumnTransformer` sin ajustar, listo para incluirse en un
        `Pipeline` de scikit-learn.
    """
    transformador_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )
    transformador_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerico", transformador_numerico, columnas_numericas),
            ("categorico", transformador_categorico, columnas_categoricas),
        ]
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    configuracion: ConfiguracionSplit = ConfiguracionSplit(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide el dataset en conjuntos de entrenamiento y prueba.

    Args:
        X: Matriz de variables predictoras.
        y: Vector de la variable objetivo.
        configuracion: Parámetros de la división (proporción,
            estratificación, semilla).

    Returns:
        Tupla `(X_train, X_test, y_train, y_test)`.
    """
    estrato = y if configuracion.estratificar else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=configuracion.proporcion_test,
        random_state=configuracion.random_state,
        stratify=estrato,
    )
    logger.info(
        "Split completado -> train: %d filas | test: %d filas (tasa positiva train=%.3f, test=%.3f)",
        len(X_train),
        len(X_test),
        y_train.mean(),
        y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Funciones: modelo y métricas (compartidas con model_training_evaluation.py)
# ---------------------------------------------------------------------
def build_model(
    estimador: BaseEstimator | ClassifierMixin,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Ensambla preprocesamiento + estimador en un único `Pipeline`.

    Empaquetar preprocesamiento y modelo en un solo objeto garantiza
    que la transformación aplicada en entrenamiento sea idéntica a la
    aplicada en inferencia (paridad train/serve), y permite serializar
    un único artefacto (`best_model.pkl`) con `joblib`.

    Args:
        estimador: Clasificador de scikit-learn (o compatible) sin
            ajustar.
        columnas_numericas: Columnas numéricas del dataset.
        columnas_categoricas: Columnas categóricas del dataset.

    Returns:
        `Pipeline` con los pasos `preprocesador` y `clasificador`.
    """
    preprocesador = create_preprocessing_pipeline(columnas_numericas, columnas_categoricas)
    return Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", estimador)])


def summarize_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula un resumen estándar de métricas de clasificación binaria.

    Centraliza el cálculo de métricas para que `model_training_evaluation.py`,
    `model_deploy.py` y `model_monitoring.py` reporten siempre las mismas
    definiciones, evitando duplicación e inconsistencias entre etapas.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas (clase dura).
        y_proba: Probabilidad predicha de la clase positiva. Si es
            ``None``, se omiten las métricas basadas en score (ROC AUC,
            PR AUC, KS).

    Returns:
        Diccionario con `accuracy`, `precision`, `recall`, `f1` y,
        cuando `y_proba` está disponible, `roc_auc`, `pr_auc` y `ks_statistic`.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metricas: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        y_true_array = np.asarray(y_true)
        puntajes_positivos = y_proba[y_true_array == 1]
        puntajes_negativos = y_proba[y_true_array == 0]
        estadistico_ks = 0.0
        if len(puntajes_positivos) and len(puntajes_negativos):
            from scipy.stats import ks_2samp

            estadistico_ks = float(ks_2samp(puntajes_positivos, puntajes_negativos).statistic)

        metricas.update(
            {
                "roc_auc": roc_auc_score(y_true, y_proba),
                "pr_auc": average_precision_score(y_true, y_proba),
                "ks_statistic": estadistico_ks,
            }
        )

    logger.info("Métricas calculadas: %s", {k: round(v, 4) for k, v in metricas.items()})
    return metricas


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    """Ejecuta el flujo completo de feature engineering de forma aislada.

    Pensado para validación manual o para ejecutarse como job
    independiente en Jenkins (smoke test del módulo antes de invocar
    `model_training_evaluation.py`).
    """
    try:
        dataframe_crudo = load_data()
        X, y, columnas_numericas, columnas_categoricas, _ = prepare_dataset(dataframe_crudo)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        preprocesador = create_preprocessing_pipeline(columnas_numericas, columnas_categoricas)
        preprocesador.fit(X_train)
        logger.info(
            "Smoke test OK. X_train=%s X_test=%s dimensión post-transform=%d",
            X_train.shape,
            X_test.shape,
            preprocesador.transform(X_train).shape[1],
        )
    except Exception:
        logger.exception("Fallo no controlado en ft_engineering.main()")
        raise


if __name__ == "__main__":
    main()
