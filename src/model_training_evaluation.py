"""Núcleo de entrenamiento del pipeline de riesgo crediticio.

Entrena, optimiza, compara y selecciona automáticamente el mejor modelo
de clasificación binaria (`Pago_atiempo`) entre cinco familias
(Regresión Logística, Random Forest, XGBoost, LightGBM, CatBoost) y dos
estrategias de ensamblado (Stacking, Blending), y serializa los
artefactos que consume `model_deploy.py` y `model_monitoring.py`.

Decisiones de diseño relevantes (documentadas también inline):

* **Métrica principal**: el target está fuertemente desbalanceado
  (~95 % positivos, ver `comprension_eda.ipynb`), por lo que la
  selección de modelos usa **PR AUC** (`average_precision`) en lugar de
  accuracy o incluso ROC AUC, que resultan optimistas bajo desbalance.
* **Partición de datos**: `X_test` (20 %) se aparta al inicio y nunca
  se usa para seleccionar el modelo campeón, solo para el reporte final
  no sesgado. Dentro de `X_train` se separa además un `X_val` (20 % de
  `X_train`) usado para *early stopping* de los modelos de boosting y
  para la optimización de pesos del blending.
* **CatBoost** usa un preprocesador propio (sin one-hot) para explotar
  su manejo nativo de variables categóricas vía `cat_features`, en
  lugar del `ColumnTransformer` genérico de `ft_engineering.py`.
* **SHAP** no está en la lista de librerías aprobadas para producción;
  se sustituye por importancia por ganancia (`gain`) y coeficientes
  estandarizados, documentado en `generate_visualizations`.
* **Nested CV** se ejecuta solo para Regresión Logística (el modelo más
  económico computacionalmente) como verificación de sesgo del proceso
  de búsqueda de hiperparámetros; aplicarlo a las cinco familias sería
  computacionalmente prohibitivo también en producción.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    validation_curve,
)
from sklearn.pipeline import Pipeline

from ft_engineering import (
    ConfiguracionSplit,
    RANDOM_STATE,
    VARIABLE_OBJETIVO,
    build_model,
    create_preprocessing_pipeline,
    load_data,
    prepare_dataset,
    split_dataset,
    summarize_classification,
)

matplotlib.use("Agg")  # Backend no interactivo: los scripts corren headless en CI/Jenkins.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_training_evaluation")

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
N_SPLITS_CV: Final[int] = 5
N_SPLITS_CV_OPTUNA: Final[int] = 3  # CV más liviano dentro de cada trial de Optuna.
N_TRIALS_OPTUNA: Final[int] = 15  # Subir a 50-100 en un entorno de entrenamiento productivo.
N_ITER_RANDOM_SEARCH: Final[int] = 8
RONDAS_EARLY_STOPPING: Final[int] = 30
METRICA_PRINCIPAL: Final[str] = "average_precision"  # PR AUC: robusta al desbalance del target.

RUTA_ARTEFACTOS: Final[Path] = Path("../model_artifacts")
RUTA_MEJOR_MODELO: Final[Path] = RUTA_ARTEFACTOS / "best_model.pkl"
RUTA_METRICAS: Final[Path] = RUTA_ARTEFACTOS / "metrics.json"
RUTA_FEATURE_IMPORTANCE: Final[Path] = RUTA_ARTEFACTOS / "feature_importance.csv"
RUTA_GRAFICOS: Final[Path] = RUTA_ARTEFACTOS / "graficos"
RUTA_ESTADISTICAS_PREPROCESAMIENTO: Final[Path] = RUTA_ARTEFACTOS / "preprocessing_stats.json"
RUTA_DATOS_REFERENCIA: Final[Path] = RUTA_ARTEFACTOS / "reference_data.csv"

GRID_LOGISTIC_REGRESSION: Final[dict[str, list[Any]]] = {
    "clasificador__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "clasificador__penalty": ["l1", "l2"],
    "clasificador__solver": ["liblinear"],
    "clasificador__class_weight": ["balanced"],
}
GRID_RANDOM_FOREST_ANGOSTO: Final[dict[str, list[Any]]] = {
    "clasificador__n_estimators": [200, 400],
    "clasificador__max_depth": [8, 16],
}
DISTRIBUCION_RANDOM_FOREST_AMPLIA: Final[dict[str, list[Any]]] = {
    "clasificador__n_estimators": [100, 200, 300, 400, 500],
    "clasificador__max_depth": [4, 8, 12, 16, None],
    "clasificador__min_samples_leaf": [1, 2, 5, 10],
    "clasificador__min_samples_split": [2, 5, 10],
    "clasificador__max_features": ["sqrt", "log2"],
}


@dataclass
class ResultadoModelo:
    """Contenedor estándar del resultado de entrenar un modelo.

    Attributes:
        nombre: Identificador legible del modelo (p. ej. ``"xgboost"``).
        pipeline: Pipeline entrenado (preprocesamiento + clasificador).
        metricas_validacion: Métricas calculadas sobre el conjunto de
            validación, usadas para la selección automática del campeón.
        mejores_parametros: Hiperparámetros ganadores de la búsqueda.
        tiempo_entrenamiento_segundos: Duración total de entrenamiento
            y optimización.
        estrategia_busqueda: Método usado para optimizar hiperparámetros
            (``"grid_search"``, ``"random_search"``, ``"optuna"``, etc.).
    """

    nombre: str
    pipeline: Any
    metricas_validacion: dict[str, float] = field(default_factory=dict)
    mejores_parametros: dict[str, Any] = field(default_factory=dict)
    tiempo_entrenamiento_segundos: float = 0.0
    estrategia_busqueda: str = ""
    pipeline_para_ensamble: Any = None
    """Variante sin *early stopping* horneado, apta para ser clonada y
    reentrenada por `StackingClassifier`/`BlendingClassifier`. Para modelos
    sin *early stopping* (Regresión Logística, Random Forest) es idéntica
    a `pipeline`."""


# =======================================================================
# Estrategias de validación cruzada
# =======================================================================
def crear_stratified_kfold(n_splits: int = N_SPLITS_CV) -> StratifiedKFold:
    """Crea un `StratifiedKFold` reproducible, apto para targets desbalanceados."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def crear_group_kfold(n_splits: int = N_SPLITS_CV) -> GroupKFold:
    """Crea un `GroupKFold` para evitar fuga de información entre folds cuando
    existe una entidad (p. ej. cliente) con múltiples registros."""
    return GroupKFold(n_splits=n_splits)


def crear_stratified_group_kfold(n_splits: int = N_SPLITS_CV) -> StratifiedGroupKFold:
    """Crea un `StratifiedGroupKFold`: combina estratificación por clase y
    separación por grupo/entidad."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def crear_time_series_split(n_splits: int = N_SPLITS_CV) -> TimeSeriesSplit:
    """Crea un `TimeSeriesSplit` para escenarios con dependencia temporal
    explícita entre observaciones."""
    return TimeSeriesSplit(n_splits=n_splits)


def select_validation_strategy(
    tiene_grupos: bool = False,
    tiene_orden_temporal: bool = False,
    n_splits: int = N_SPLITS_CV,
) -> tuple[Any, str]:
    """Selecciona automáticamente la estrategia de validación más adecuada.

    Reglas de decisión (documentadas para auditoría del pipeline):
        1. Si existe dependencia temporal relevante para el negocio →
           `TimeSeriesSplit` (evita usar el futuro para predecir el pasado).
        2. Si existe una entidad repetida (p. ej. mismo cliente con
           varios créditos) → `StratifiedGroupKFold` (evita fuga de
           información de un mismo cliente entre train y validación).
        3. En cualquier otro caso → `StratifiedKFold`, apropiado para el
           dataset actual: cada fila es un crédito originado de forma
           independiente (sin identificador de cliente repetido) y el
           target está desbalanceado, por lo que la estratificación es
           indispensable.

    Args:
        tiene_grupos: Indica si el dataset tiene un identificador de
            grupo/entidad que deba respetarse entre folds.
        tiene_orden_temporal: Indica si la validación debe respetar el
            orden cronológico de las observaciones.
        n_splits: Número de particiones de la validación cruzada.

    Returns:
        Tupla `(objeto_cv, nombre_estrategia)`.
    """
    if tiene_orden_temporal:
        cv, nombre = crear_time_series_split(n_splits), "TimeSeriesSplit"
    elif tiene_grupos:
        cv, nombre = crear_stratified_group_kfold(n_splits), "StratifiedGroupKFold"
    else:
        cv, nombre = crear_stratified_kfold(n_splits), "StratifiedKFold"

    logger.info(
        "Estrategia de validación seleccionada: %s (grupos=%s, temporal=%s)",
        nombre,
        tiene_grupos,
        tiene_orden_temporal,
    )
    return cv, nombre


# =======================================================================
# Búsqueda de hiperparámetros (genérica, reutilizable por cualquier modelo)
# =======================================================================
def run_grid_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = METRICA_PRINCIPAL,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """Ejecuta `GridSearchCV` y retorna el mejor estimador ya ajustado.

    Args:
        pipeline: Pipeline base (preprocesador + clasificador) sin ajustar.
        param_grid: Grilla exhaustiva de hiperparámetros a evaluar.
        X: Variables predictoras de entrenamiento.
        y: Variable objetivo de entrenamiento.
        cv: Estrategia de validación cruzada.
        scoring: Métrica objetivo de la búsqueda.

    Returns:
        Tupla `(mejor_estimador, mejores_parametros, mejor_score, tiempo_segundos)`.
    """
    inicio = time.perf_counter()
    buscador = GridSearchCV(
        pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True
    )
    buscador.fit(X, y)
    tiempo_segundos = time.perf_counter() - inicio
    logger.info(
        "GridSearchCV completado en %.1fs | mejor %s=%.4f | params=%s",
        tiempo_segundos,
        scoring,
        buscador.best_score_,
        buscador.best_params_,
    )
    return buscador.best_estimator_, buscador.best_params_, buscador.best_score_, tiempo_segundos


def run_random_search(
    pipeline: Pipeline,
    param_distributions: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    scoring: str = METRICA_PRINCIPAL,
    n_iter: int = N_ITER_RANDOM_SEARCH,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """Ejecuta `RandomizedSearchCV` y retorna el mejor estimador ya ajustado.

    Args:
        pipeline: Pipeline base (preprocesador + clasificador) sin ajustar.
        param_distributions: Espacio de hiperparámetros a muestrear.
        X: Variables predictoras de entrenamiento.
        y: Variable objetivo de entrenamiento.
        cv: Estrategia de validación cruzada.
        scoring: Métrica objetivo de la búsqueda.
        n_iter: Número de combinaciones aleatorias a evaluar.

    Returns:
        Tupla `(mejor_estimador, mejores_parametros, mejor_score, tiempo_segundos)`.
    """
    inicio = time.perf_counter()
    buscador = RandomizedSearchCV(
        pipeline,
        param_distributions,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
        refit=True,
    )
    buscador.fit(X, y)
    tiempo_segundos = time.perf_counter() - inicio
    logger.info(
        "RandomizedSearchCV completado en %.1fs | mejor %s=%.4f | params=%s",
        tiempo_segundos,
        scoring,
        buscador.best_score_,
        buscador.best_params_,
    )
    return buscador.best_estimator_, buscador.best_params_, buscador.best_score_, tiempo_segundos


def run_optuna_search(
    nombre_modelo: str,
    funcion_objetivo: Callable[[optuna.Trial], float],
    n_trials: int = N_TRIALS_OPTUNA,
) -> tuple[dict[str, Any], float, float]:
    """Ejecuta una optimización bayesiana de hiperparámetros con Optuna.

    Args:
        nombre_modelo: Nombre del modelo, usado únicamente para logging.
        funcion_objetivo: Función objetivo `Trial -> score` a maximizar.
        n_trials: Número de combinaciones a explorar.

    Returns:
        Tupla `(mejores_parametros, mejor_score, tiempo_segundos)`.
    """
    inicio = time.perf_counter()
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    estudio.optimize(funcion_objetivo, n_trials=n_trials, show_progress_bar=False)
    tiempo_segundos = time.perf_counter() - inicio
    logger.info(
        "Optuna (%s) completado en %.1fs tras %d trials | mejor %s=%.4f",
        nombre_modelo,
        tiempo_segundos,
        n_trials,
        METRICA_PRINCIPAL,
        estudio.best_value,
    )
    return estudio.best_params, estudio.best_value, tiempo_segundos


def run_nested_cross_validation(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_externo: Any,
    cv_interno: Any,
    scoring: str = METRICA_PRINCIPAL,
) -> tuple[np.ndarray, float, float]:
    """Ejecuta validación cruzada anidada como verificación de sesgo optimista.

    El bucle externo estima el desempeño de generalización del *proceso*
    de selección de hiperparámetros (no de una configuración fija),
    evitando el sesgo optimista de reportar el score de la búsqueda
    interna directamente como desempeño esperado en producción.

    Args:
        pipeline: Pipeline base (preprocesador + clasificador) sin ajustar.
        param_grid: Grilla de hiperparámetros para la búsqueda interna.
        X: Variables predictoras.
        y: Variable objetivo.
        cv_externo: Validación cruzada externa (estima generalización).
        cv_interno: Validación cruzada interna (selecciona hiperparámetros).
        scoring: Métrica de evaluación.

    Returns:
        Tupla `(scores_por_fold, media, desviación_estándar)`.
    """
    scores: list[float] = []
    for indice_fold, (idx_train, idx_val) in enumerate(cv_externo.split(X, y), start=1):
        X_tr, X_val = X.iloc[idx_train], X.iloc[idx_val]
        y_tr, y_val = y.iloc[idx_train], y.iloc[idx_val]

        buscador = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv_interno, n_jobs=-1)
        buscador.fit(X_tr, y_tr)
        probabilidades = buscador.predict_proba(X_val)[:, 1]
        score_fold = average_precision_score(y_val, probabilidades)
        scores.append(score_fold)
        logger.info("Nested CV fold %d/%d: %s=%.4f", indice_fold, cv_externo.get_n_splits(), scoring, score_fold)

    scores_array = np.array(scores)
    logger.info(
        "Nested CV finalizado: %s medio=%.4f (+/- %.4f)",
        scoring,
        scores_array.mean(),
        scores_array.std(),
    )
    return scores_array, float(scores_array.mean()), float(scores_array.std())


def compare_search_strategies(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Construye una tabla comparativa de tiempos y resultados de búsqueda.

    Args:
        resultados: Diccionario `{nombre_estrategia: {"tiempo_segundos": ..,
            "mejor_score": ..}}`.

    Returns:
        DataFrame ordenado descendentemente por `mejor_score`.
    """
    tabla = pd.DataFrame(resultados).T
    tabla = tabla.sort_values("mejor_score", ascending=False)
    logger.info("Comparación de estrategias de búsqueda:\n%s", tabla.to_string())
    return tabla


# =======================================================================
# Espacios de búsqueda Optuna por familia de modelo
# =======================================================================
def _construir_objetivo_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Callable[[optuna.Trial], float]:
    """Crea la función objetivo de Optuna para XGBoost."""

    def objetivo(trial: optuna.Trial) -> float:
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        estimador = xgb.XGBClassifier(
            **parametros,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            n_jobs=-1,
        )
        pipeline = build_model(estimador, columnas_numericas, columnas_categoricas)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=-1)
        return float(scores.mean())

    return objetivo


def _construir_objetivo_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Callable[[optuna.Trial], float]:
    """Crea la función objetivo de Optuna para LightGBM."""

    def objetivo(trial: optuna.Trial) -> float:
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }
        estimador = lgb.LGBMClassifier(
            **parametros,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
        )
        pipeline = build_model(estimador, columnas_numericas, columnas_categoricas)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=-1)
        return float(scores.mean())

    return objetivo


def create_preprocessing_pipeline_catboost(
    columnas_numericas: list[str], columnas_categoricas: list[str]
) -> ColumnTransformer:
    """Construye un preprocesador liviano para CatBoost.

    A diferencia de `ft_engineering.create_preprocessing_pipeline`, no
    aplica one-hot encoding: solo imputa nulos y conserva los nombres de
    columna originales (`verbose_feature_names_out=False`), permitiendo
    que CatBoost reciba las categóricas en texto plano y las gestione de
    forma nativa vía `cat_features`, que es su principal ventaja frente
    a los demás modelos del proyecto.

    Args:
        columnas_numericas: Nombres de las columnas numéricas.
        columnas_categoricas: Nombres de las columnas categóricas.

    Returns:
        `ColumnTransformer` configurado con salida en formato `pandas`.
    """
    transformador = ColumnTransformer(
        transformers=[
            ("numerico", SimpleImputer(strategy="median"), columnas_numericas),
            ("categorico", SimpleImputer(strategy="most_frequent"), columnas_categoricas),
        ],
        verbose_feature_names_out=False,
    )
    return transformador.set_output(transform="pandas")


class CatBoostWrapperClassifier(ClassifierMixin, BaseEstimator):
    """Envoltorio compatible con scikit-learn para `CatBoostClassifier`.

    `CatBoostClassifier` no es clonable por `sklearn.base.clone` cuando se
    le pasa `cat_features` en el constructor: internamente transforma esa
    lista y la identidad del objeto ya no coincide con la original, lo que
    hace que `__sklearn_clone__` levante un `RuntimeError` (limitación
    conocida de la librería). Esto rompe cualquier mecanismo de
    scikit-learn que dependa de clonar el estimador —`cross_val_score`,
    `GridSearchCV`, y especialmente `StackingClassifier`, que clona y
    reentrena cada learner base por fold para generar predicciones
    out-of-fold—.

    Este envoltorio resuelve el problema de raíz: al ser un
    `BaseEstimator` propio que solo almacena sus parámetros como
    atributos sin transformarlos (convención estándar de scikit-learn),
    `clone()` funciona de forma nativa. El objeto `CatBoostClassifier`
    real se construye recién dentro de `fit`.

    Attributes:
        columnas_categoricas: Nombres de columnas a tratar como
            categóricas nativas (pasadas a `cat_features` en `fit`).
    """

    def __init__(
        self,
        columnas_categoricas: list[str] | None = None,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_state: int = RANDOM_STATE,
        auto_class_weights: str = "Balanced",
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.columnas_categoricas = columnas_categoricas
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.auto_class_weights = auto_class_weights
        self.early_stopping_rounds = early_stopping_rounds

    def fit(
        self, X: pd.DataFrame, y: pd.Series, eval_set: tuple[Any, pd.Series] | None = None
    ) -> "CatBoostWrapperClassifier":
        """Entrena el `CatBoostClassifier` interno, opcionalmente con early stopping."""
        self.modelo_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            cat_features=list(self.columnas_categoricas or []),
            random_state=self.random_state,
            auto_class_weights=self.auto_class_weights,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )
        if eval_set is not None:
            self.modelo_.fit(X, y, eval_set=eval_set)
        else:
            self.modelo_.fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Delegado directo al `CatBoostClassifier` interno ya ajustado."""
        return self.modelo_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Delegado directo al `CatBoostClassifier` interno ya ajustado."""
        return self.modelo_.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Expone la importancia de variables del modelo interno."""
        return self.modelo_.get_feature_importance()


def build_catboost_pipeline(
    parametros: dict[str, Any], columnas_numericas: list[str], columnas_categoricas: list[str]
) -> Pipeline:
    """Ensambla el pipeline específico de CatBoost con manejo nativo de categóricas.

    Args:
        parametros: Hiperparámetros de `CatBoostWrapperClassifier`.
        columnas_numericas: Columnas numéricas del dataset.
        columnas_categoricas: Columnas categóricas del dataset.

    Returns:
        `Pipeline` con preprocesador liviano + `CatBoostWrapperClassifier`.
    """
    preprocesador = create_preprocessing_pipeline_catboost(columnas_numericas, columnas_categoricas)
    clasificador = CatBoostWrapperClassifier(columnas_categoricas=columnas_categoricas, **parametros)
    return Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", clasificador)])


def _construir_objetivo_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Callable[[optuna.Trial], float]:
    """Crea la función objetivo de Optuna para CatBoost."""

    def objetivo(trial: optuna.Trial) -> float:
        parametros = {
            "iterations": trial.suggest_int("iterations", 100, 400),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
        pipeline = build_catboost_pipeline(parametros, columnas_numericas, columnas_categoricas)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=1)
        return float(scores.mean())

    return objetivo


# =======================================================================
# Entrenamiento con early stopping (modelos de boosting)
# =======================================================================
def _transformar_validacion(
    preprocesador: ColumnTransformer, X_tr: pd.DataFrame, X_val: pd.DataFrame
) -> Any:
    """Ajusta un preprocesador en train y transforma el conjunto de validación."""
    preprocesador.fit(X_tr)
    return preprocesador.transform(X_val)


def entrenar_xgboost_con_early_stopping(
    mejores_parametros: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena XGBoost con los mejores hiperparámetros usando early stopping."""
    preprocesador = create_preprocessing_pipeline(columnas_numericas, columnas_categoricas)
    X_val_transformado = _transformar_validacion(preprocesador, X_tr, X_val)

    clasificador = xgb.XGBClassifier(
        **mejores_parametros,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        early_stopping_rounds=RONDAS_EARLY_STOPPING,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", clasificador)])
    pipeline.fit(
        X_tr,
        y_tr,
        clasificador__eval_set=[(X_val_transformado, y_val)],
        clasificador__verbose=False,
    )
    mejor_iteracion = int(clasificador.best_iteration)
    logger.info("XGBoost: early stopping detuvo en la iteración %d", mejor_iteracion)
    return pipeline, mejor_iteracion


def construir_pipeline_xgboost_para_ensamble(
    mejores_parametros: dict[str, Any],
    mejor_iteracion: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena XGBoost sin early stopping, fijando `n_estimators` al valor
    óptimo hallado, para poder usarse como *base learner* de un ensamble
    (Stacking/Blending necesitan clonar y reentrenar el estimador sin
    depender de un `eval_set` externo)."""
    parametros_finales = {**mejores_parametros, "n_estimators": max(mejor_iteracion, 1)}
    estimador = xgb.XGBClassifier(
        **parametros_finales, random_state=RANDOM_STATE, eval_metric="aucpr", n_jobs=-1
    )
    pipeline = build_model(estimador, columnas_numericas, columnas_categoricas)
    pipeline.fit(X_tr, y_tr)
    return pipeline


def entrenar_lightgbm_con_early_stopping(
    mejores_parametros: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena LightGBM con los mejores hiperparámetros usando early stopping."""
    preprocesador = create_preprocessing_pipeline(columnas_numericas, columnas_categoricas)
    X_val_transformado = _transformar_validacion(preprocesador, X_tr, X_val)

    clasificador = lgb.LGBMClassifier(
        **mejores_parametros,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", clasificador)])
    pipeline.fit(
        X_tr,
        y_tr,
        clasificador__eval_set=[(X_val_transformado, y_val)],
        clasificador__callbacks=[lgb.early_stopping(RONDAS_EARLY_STOPPING, verbose=False)],
    )
    mejor_iteracion = int(clasificador.best_iteration_)
    logger.info("LightGBM: early stopping detuvo en la iteración %d", mejor_iteracion)
    return pipeline, mejor_iteracion


def construir_pipeline_lightgbm_para_ensamble(
    mejores_parametros: dict[str, Any],
    mejor_iteracion: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena LightGBM sin early stopping, fijando `n_estimators` al valor
    óptimo hallado, para poder usarse como *base learner* de un ensamble."""
    parametros_finales = {**mejores_parametros, "n_estimators": max(mejor_iteracion, 1)}
    estimador = lgb.LGBMClassifier(
        **parametros_finales, random_state=RANDOM_STATE, class_weight="balanced", verbosity=-1, n_jobs=-1
    )
    pipeline = build_model(estimador, columnas_numericas, columnas_categoricas)
    pipeline.fit(X_tr, y_tr)
    return pipeline


def entrenar_catboost_con_early_stopping(
    mejores_parametros: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena CatBoost con los mejores hiperparámetros usando early stopping."""
    preprocesador = create_preprocessing_pipeline_catboost(columnas_numericas, columnas_categoricas)
    X_val_transformado = _transformar_validacion(preprocesador, X_tr, X_val)

    clasificador = CatBoostWrapperClassifier(
        columnas_categoricas=columnas_categoricas,
        **mejores_parametros,
        early_stopping_rounds=RONDAS_EARLY_STOPPING,
    )
    pipeline = Pipeline(steps=[("preprocesador", preprocesador), ("clasificador", clasificador)])
    pipeline.fit(X_tr, y_tr, clasificador__eval_set=(X_val_transformado, y_val))
    mejor_iteracion = int(clasificador.modelo_.best_iteration_)
    logger.info("CatBoost: early stopping detuvo en la iteración %d", mejor_iteracion)
    return pipeline, mejor_iteracion


def construir_pipeline_catboost_para_ensamble(
    mejores_parametros: dict[str, Any],
    mejor_iteracion: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> Pipeline:
    """Reentrena CatBoost sin early stopping, fijando `iterations` al valor
    óptimo hallado, para poder usarse como *base learner* de un ensamble."""
    parametros_finales = {**mejores_parametros, "iterations": max(mejor_iteracion, 1)}
    pipeline = build_catboost_pipeline(parametros_finales, columnas_numericas, columnas_categoricas)
    pipeline.fit(X_tr, y_tr)
    return pipeline


# =======================================================================
# Entrenamiento por familia de modelo
# =======================================================================
def train_logistic_regression(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cv: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> ResultadoModelo:
    """Entrena Regresión Logística regularizada con búsqueda en grilla.

    Incluye penalización L1/L2, escalado (heredado del preprocesador
    estándar) y ponderación de clases para compensar el desbalance.
    """
    inicio = time.perf_counter()
    pipeline_base = build_model(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        columnas_numericas,
        columnas_categoricas,
    )
    mejor_estimador, mejores_parametros, mejor_score, _ = run_grid_search(
        pipeline_base, GRID_LOGISTIC_REGRESSION, X_tr, y_tr, cv
    )
    tiempo_total = time.perf_counter() - inicio
    return ResultadoModelo(
        nombre="logistic_regression",
        pipeline=mejor_estimador,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": mejor_score},
        mejores_parametros=mejores_parametros,
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="grid_search",
    )


def train_random_forest(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cv: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> ResultadoModelo:
    """Entrena Random Forest comparando Grid Search vs. Randomized Search.

    Ambas estrategias se ejecutan sobre el mismo pipeline base y se
    conserva la que obtiene mejor `average_precision`, dejando registro
    de tiempos y resultados de ambas (ver logs de `compare_search_strategies`).
    """
    pipeline_base = build_model(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        columnas_numericas,
        columnas_categoricas,
    )

    estimador_grid, params_grid, score_grid, tiempo_grid = run_grid_search(
        pipeline_base, GRID_RANDOM_FOREST_ANGOSTO, X_tr, y_tr, cv
    )
    estimador_random, params_random, score_random, tiempo_random = run_random_search(
        pipeline_base, DISTRIBUCION_RANDOM_FOREST_AMPLIA, X_tr, y_tr, cv
    )

    tabla_comparativa = compare_search_strategies(
        {
            "grid_search": {"tiempo_segundos": tiempo_grid, "mejor_score": score_grid},
            "random_search": {"tiempo_segundos": tiempo_random, "mejor_score": score_random},
        }
    )
    tabla_comparativa.to_csv(RUTA_ARTEFACTOS / "rf_comparacion_busqueda.csv")

    if score_random >= score_grid:
        mejor_estimador, mejores_parametros, mejor_score = estimador_random, params_random, score_random
        estrategia, tiempo_total = "random_search", tiempo_random
    else:
        mejor_estimador, mejores_parametros, mejor_score = estimador_grid, params_grid, score_grid
        estrategia, tiempo_total = "grid_search", tiempo_grid

    return ResultadoModelo(
        nombre="random_forest",
        pipeline=mejor_estimador,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": mejor_score},
        mejores_parametros=mejores_parametros,
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda=estrategia,
    )


def train_xgboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_optuna: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> ResultadoModelo:
    """Optimiza XGBoost con Optuna y reentrena con early stopping."""
    objetivo = _construir_objetivo_xgboost(X_tr, y_tr, cv_optuna, columnas_numericas, columnas_categoricas)
    mejores_parametros, mejor_score, tiempo_busqueda = run_optuna_search("xgboost", objetivo)

    inicio_refit = time.perf_counter()
    pipeline_final, mejor_iteracion = entrenar_xgboost_con_early_stopping(
        mejores_parametros, X_tr, y_tr, X_val, y_val, columnas_numericas, columnas_categoricas
    )
    pipeline_ensamble = construir_pipeline_xgboost_para_ensamble(
        mejores_parametros, mejor_iteracion, X_tr, y_tr, columnas_numericas, columnas_categoricas
    )
    tiempo_total = tiempo_busqueda + (time.perf_counter() - inicio_refit)

    return ResultadoModelo(
        nombre="xgboost",
        pipeline=pipeline_final,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": mejor_score},
        mejores_parametros=mejores_parametros,
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="optuna",
        pipeline_para_ensamble=pipeline_ensamble,
    )


def train_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_optuna: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> ResultadoModelo:
    """Optimiza LightGBM con Optuna y reentrena con early stopping."""
    objetivo = _construir_objetivo_lightgbm(X_tr, y_tr, cv_optuna, columnas_numericas, columnas_categoricas)
    mejores_parametros, mejor_score, tiempo_busqueda = run_optuna_search("lightgbm", objetivo)

    inicio_refit = time.perf_counter()
    pipeline_final, mejor_iteracion = entrenar_lightgbm_con_early_stopping(
        mejores_parametros, X_tr, y_tr, X_val, y_val, columnas_numericas, columnas_categoricas
    )
    pipeline_ensamble = construir_pipeline_lightgbm_para_ensamble(
        mejores_parametros, mejor_iteracion, X_tr, y_tr, columnas_numericas, columnas_categoricas
    )
    tiempo_total = tiempo_busqueda + (time.perf_counter() - inicio_refit)

    return ResultadoModelo(
        nombre="lightgbm",
        pipeline=pipeline_final,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": mejor_score},
        mejores_parametros=mejores_parametros,
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="optuna",
        pipeline_para_ensamble=pipeline_ensamble,
    )


def train_catboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_optuna: Any,
    columnas_numericas: list[str],
    columnas_categoricas: list[str],
) -> ResultadoModelo:
    """Optimiza CatBoost con Optuna y reentrena con early stopping."""
    objetivo = _construir_objetivo_catboost(X_tr, y_tr, cv_optuna, columnas_numericas, columnas_categoricas)
    mejores_parametros, mejor_score, tiempo_busqueda = run_optuna_search("catboost", objetivo)

    inicio_refit = time.perf_counter()
    pipeline_final, mejor_iteracion = entrenar_catboost_con_early_stopping(
        mejores_parametros, X_tr, y_tr, X_val, y_val, columnas_numericas, columnas_categoricas
    )
    pipeline_ensamble = construir_pipeline_catboost_para_ensamble(
        mejores_parametros, mejor_iteracion, X_tr, y_tr, columnas_numericas, columnas_categoricas
    )
    tiempo_total = tiempo_busqueda + (time.perf_counter() - inicio_refit)

    return ResultadoModelo(
        nombre="catboost",
        pipeline=pipeline_final,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": mejor_score},
        mejores_parametros=mejores_parametros,
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="optuna",
        pipeline_para_ensamble=pipeline_ensamble,
    )


# =======================================================================
# Ensamblado: Stacking y Blending
# =======================================================================
def build_stacking_ensemble(
    estimadores_base: list[tuple[str, Any]], meta_learner: BaseEstimator, cv: Any
) -> StackingClassifier:
    """Construye un `StackingClassifier` sin ajustar.

    Args:
        estimadores_base: Lista `(nombre, pipeline_sin_ajustar)` de los
            modelos base (XGBoost, LightGBM, CatBoost).
        meta_learner: Estimador que combina las predicciones base
            (Regresión Logística, según especificación del proyecto).
        cv: Estrategia de validación cruzada usada internamente por
            `StackingClassifier` para generar predicciones out-of-fold
            y evitar fuga de información hacia el meta-modelo.

    Returns:
        `StackingClassifier` listo para `.fit()`.
    """
    return StackingClassifier(
        estimators=estimadores_base,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def train_stacking(
    estimadores_base: list[tuple[str, Any]], X_tr: pd.DataFrame, y_tr: pd.Series, cv: Any
) -> ResultadoModelo:
    """Entrena el ensamble de Stacking y mide su desempeño por validación cruzada."""
    inicio = time.perf_counter()
    meta_learner = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    stacking = build_stacking_ensemble(estimadores_base, meta_learner, cv)

    scores_cv = cross_val_score(stacking, X_tr, y_tr, cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=-1)
    stacking.fit(X_tr, y_tr)
    tiempo_total = time.perf_counter() - inicio

    logger.info("Stacking: %s_cv=%.4f (+/- %.4f)", METRICA_PRINCIPAL, scores_cv.mean(), scores_cv.std())
    return ResultadoModelo(
        nombre="stacking",
        pipeline=stacking,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": float(scores_cv.mean())},
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="stacking_cv",
    )


class BlendingClassifier(ClassifierMixin, BaseEstimator):
    """Ensamble por Blending: combina modelos base ya entrenados mediante
    un promedio ponderado de probabilidades, con pesos optimizados sobre
    un conjunto de holdout independiente del entrenamiento de cada base.

    Attributes:
        estimadores_base: Lista `(nombre, pipeline_ya_entrenado)`.
        pesos_: Pesos óptimos aprendidos en `fit` (uno por estimador base).
    """

    def __init__(self, estimadores_base: list[tuple[str, Any]]) -> None:
        self.estimadores_base = estimadores_base
        self.pesos_: np.ndarray | None = None

    def fit(self, X_holdout: pd.DataFrame, y_holdout: pd.Series) -> "BlendingClassifier":
        """Optimiza los pesos de combinación sobre el conjunto de holdout.

        Args:
            X_holdout: Variables predictoras del conjunto de holdout
                (independiente de los datos usados para entrenar cada
                modelo base).
            y_holdout: Variable objetivo del conjunto de holdout.

        Returns:
            La instancia ajustada (`self`), por convención de scikit-learn.
        """
        matriz_probabilidades = np.column_stack(
            [pipeline.predict_proba(X_holdout)[:, 1] for _, pipeline in self.estimadores_base]
        )

        def negativo_pr_auc(pesos: np.ndarray) -> float:
            probabilidad_combinada = matriz_probabilidades @ pesos
            return -average_precision_score(y_holdout, probabilidad_combinada)

        n_estimadores = len(self.estimadores_base)
        pesos_iniciales = np.full(n_estimadores, 1.0 / n_estimadores)
        restricciones = {"type": "eq", "fun": lambda pesos: np.sum(pesos) - 1.0}
        limites = [(0.0, 1.0)] * n_estimadores

        resultado = minimize(
            negativo_pr_auc,
            pesos_iniciales,
            method="SLSQP",
            bounds=limites,
            constraints=[restricciones],
        )
        self.pesos_ = resultado.x
        logger.info(
            "Blending: pesos óptimos %s",
            {nombre: round(peso, 3) for (nombre, _), peso in zip(self.estimadores_base, self.pesos_)},
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Combina las probabilidades de los modelos base según `self.pesos_`."""
        if self.pesos_ is None:
            raise RuntimeError("BlendingClassifier no ha sido ajustado. Llame a `fit` primero.")
        matriz_probabilidades = np.column_stack(
            [pipeline.predict_proba(X)[:, 1] for _, pipeline in self.estimadores_base]
        )
        probabilidad_positiva = matriz_probabilidades @ self.pesos_
        return np.column_stack([1.0 - probabilidad_positiva, probabilidad_positiva])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la clase dura usando un umbral de decisión de 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_blending(
    estimadores_base_entrenados: list[tuple[str, Any]], X_holdout: pd.DataFrame, y_holdout: pd.Series
) -> ResultadoModelo:
    """Entrena el ensamble de Blending sobre un conjunto de holdout.

    Args:
        estimadores_base_entrenados: Modelos base ya ajustados (mismos
            pipelines de boosting usados en el resto del pipeline).
        X_holdout: Conjunto de holdout para optimizar pesos.
        y_holdout: Target del conjunto de holdout.

    Returns:
        `ResultadoModelo` con el `BlendingClassifier` ajustado.
    """
    inicio = time.perf_counter()
    blending = BlendingClassifier(estimadores_base_entrenados)
    blending.fit(X_holdout, y_holdout)
    score_holdout = average_precision_score(y_holdout, blending.predict_proba(X_holdout)[:, 1])
    tiempo_total = time.perf_counter() - inicio

    return ResultadoModelo(
        nombre="blending",
        pipeline=blending,
        metricas_validacion={f"{METRICA_PRINCIPAL}_cv": score_holdout},
        tiempo_entrenamiento_segundos=tiempo_total,
        estrategia_busqueda="blending_holdout",
    )


# =======================================================================
# Métricas de negocio, importancia de variables y visualizaciones
# =======================================================================
def calculate_lift_gain(y_true: pd.Series, y_proba: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    """Calcula la tabla de Lift y Ganancia acumulada por decil de score.

    Args:
        y_true: Etiquetas reales.
        y_proba: Probabilidad predicha de la clase positiva.
        n_deciles: Número de segmentos de score a construir.

    Returns:
        DataFrame con columnas `decil`, `tasa_positivos`, `ganancia_acumulada`
        y `lift`, ordenado del decil de mayor score al de menor.
    """
    tabla = pd.DataFrame({"y_true": np.asarray(y_true), "y_proba": y_proba})
    tabla["decil"] = pd.qcut(tabla["y_proba"], n_deciles, labels=False, duplicates="drop")
    tasa_base = tabla["y_true"].mean()

    resumen = (
        tabla.groupby("decil", observed=True)
        .agg(n_observaciones=("y_true", "size"), n_positivos=("y_true", "sum"))
        .sort_index(ascending=False)
        .reset_index()
    )
    resumen["tasa_positivos"] = resumen["n_positivos"] / resumen["n_observaciones"]
    resumen["ganancia_acumulada"] = resumen["n_positivos"].cumsum() / tabla["y_true"].sum()
    resumen["lift"] = resumen["tasa_positivos"] / tasa_base
    return resumen


def calculate_business_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
    """Resume las métricas de negocio clave (lift y ganancia) en un dict plano.

    Args:
        y_true: Etiquetas reales.
        y_proba: Probabilidad predicha de la clase positiva.

    Returns:
        Diccionario con `lift_top_decil` y `ganancia_top_20pct`.
    """
    tabla_lift_gain = calculate_lift_gain(y_true, y_proba)
    return {
        "lift_top_decil": float(tabla_lift_gain.iloc[0]["lift"]),
        "ganancia_top_20pct": float(tabla_lift_gain.iloc[:2]["n_positivos"].sum() / max(np.asarray(y_true).sum(), 1)),
    }


def _obtener_nombres_columnas_transformadas(pipeline: Any) -> list[str]:
    """Extrae los nombres de columnas post-preprocesamiento de un pipeline."""
    preprocesador = pipeline.named_steps["preprocesador"]
    return list(preprocesador.get_feature_names_out())


def extract_feature_importance(resultado: ResultadoModelo) -> pd.DataFrame:
    """Extrae la importancia de variables de un modelo entrenado.

    Soporta modelos lineales (coeficientes en valor absoluto), modelos
    de árbol/boosting con `feature_importances_` (ganancia) y CatBoost
    (`get_feature_importance`). Los ensambles (Stacking/Blending) no
    tienen una importancia nativa única y se excluyen explícitamente.

    Args:
        resultado: Resultado de entrenamiento de un modelo individual.

    Returns:
        DataFrame con columnas `modelo`, `variable`, `importancia`,
        ordenado descendentemente. Vacío si el modelo es un ensamble.
    """
    pipeline = resultado.pipeline
    if resultado.nombre in {"stacking", "blending"}:
        logger.info("Importancia de variables no aplica a ensambles (%s).", resultado.nombre)
        return pd.DataFrame(columns=["modelo", "variable", "importancia"])

    clasificador = pipeline.named_steps["clasificador"]
    nombres_columnas = _obtener_nombres_columnas_transformadas(pipeline)

    if hasattr(clasificador, "coef_"):
        importancias = np.abs(clasificador.coef_).ravel()
    elif isinstance(clasificador, CatBoostWrapperClassifier):
        importancias = clasificador.get_feature_importance()
    elif hasattr(clasificador, "feature_importances_"):
        importancias = clasificador.feature_importances_
    else:
        logger.warning("Modelo %s no expone importancia de variables.", resultado.nombre)
        return pd.DataFrame(columns=["modelo", "variable", "importancia"])

    tabla = pd.DataFrame(
        {"modelo": resultado.nombre, "variable": nombres_columnas, "importancia": importancias}
    )
    return tabla.sort_values("importancia", ascending=False).reset_index(drop=True)


def generate_visualizations(
    resultado: ResultadoModelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    directorio_salida: Path = RUTA_GRAFICOS,
) -> None:
    """Genera y guarda en disco las visualizaciones estándar del modelo campeón.

    Produce: curva ROC, curva Precision-Recall, matriz de confusión,
    importancia de variables (top 15) y curva de aprendizaje. No se
    generan SHAP values: la librería `shap` no forma parte del stack de
    producción aprobado para este proyecto; la importancia por ganancia
    ya reportada cumple el mismo propósito de interpretabilidad.

    Args:
        resultado: Resultado de entrenamiento del modelo a graficar.
        X_test: Variables predictoras de prueba (holdout final).
        y_test: Variable objetivo de prueba.
        directorio_salida: Carpeta donde se guardan los archivos `.png`.
    """
    directorio_salida.mkdir(parents=True, exist_ok=True)
    pipeline = resultado.pipeline

    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title(f"Curva ROC — {resultado.nombre}")
        fig.savefig(directorio_salida / "roc_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title(f"Curva Precision-Recall — {resultado.nombre}")
        fig.savefig(directorio_salida / "precision_recall_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax, colorbar=False)
        ax.set_title(f"Matriz de confusión — {resultado.nombre}")
        fig.savefig(directorio_salida / "confusion_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        tabla_importancia = extract_feature_importance(resultado)
        if not tabla_importancia.empty:
            top_variables = tabla_importancia.head(15).sort_values("importancia")
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(top_variables["variable"], top_variables["importancia"])
            ax.set_title(f"Importancia de variables (top 15) — {resultado.nombre}")
            fig.savefig(directorio_salida / "feature_importance.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        logger.info("Visualizaciones guardadas en %s", directorio_salida)
    except Exception:
        logger.exception("No fue posible generar una o más visualizaciones del modelo campeón.")


def _obtener_parametro_para_curva_validacion(nombre_modelo: str) -> tuple[str, list[Any]]:
    """Mapea cada familia de modelo a un hiperparámetro representativo
    (y su rango) para graficar la curva de validación.

    Args:
        nombre_modelo: Nombre del modelo individual (nunca un ensamble).

    Returns:
        Tupla `(nombre_parametro, rango_valores)` en el formato esperado
        por `sklearn.model_selection.validation_curve`.
    """
    parametros_por_modelo: dict[str, tuple[str, list[Any]]] = {
        "logistic_regression": ("clasificador__C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        "random_forest": ("clasificador__max_depth", [4, 8, 12, 16, None]),
        "xgboost": ("clasificador__max_depth", [3, 5, 7, 9]),
        "lightgbm": ("clasificador__num_leaves", [15, 31, 63, 127]),
        "catboost": ("clasificador__depth", [4, 6, 8, 10]),
    }
    return parametros_por_modelo.get(nombre_modelo, ("clasificador__max_depth", [4, 8, 12]))


def generate_learning_and_validation_curves(
    pipeline_base: Pipeline,
    param_name: str,
    param_range: list[Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    directorio_salida: Path = RUTA_GRAFICOS,
) -> None:
    """Genera y guarda la curva de aprendizaje y la curva de validación del campeón.

    Args:
        pipeline_base: Pipeline sin ajustar del modelo campeón (mismos
            hiperparámetros óptimos, salvo el que varía en la curva de
            validación).
        param_name: Nombre del hiperparámetro a variar (formato
            `"clasificador__nombre_parametro"`).
        param_range: Valores a evaluar para `param_name`.
        X: Variables predictoras (se recomienda usar solo `X_train`).
        y: Variable objetivo.
        cv: Estrategia de validación cruzada.
        directorio_salida: Carpeta donde se guardan los archivos `.png`.
    """
    directorio_salida.mkdir(parents=True, exist_ok=True)
    try:
        tamanos_train, scores_train, scores_val = learning_curve(
            pipeline_base, X, y, cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5),
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tamanos_train, scores_train.mean(axis=1), marker="o", label="Entrenamiento")
        ax.plot(tamanos_train, scores_val.mean(axis=1), marker="o", label="Validación")
        ax.set_xlabel("Tamaño del conjunto de entrenamiento")
        ax.set_ylabel(METRICA_PRINCIPAL)
        ax.set_title("Curva de aprendizaje")
        ax.legend()
        fig.savefig(directorio_salida / "learning_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        scores_train_val, scores_val_val = validation_curve(
            pipeline_base, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=METRICA_PRINCIPAL, n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(param_range, scores_train_val.mean(axis=1), marker="o", label="Entrenamiento")
        ax.plot(param_range, scores_val_val.mean(axis=1), marker="o", label="Validación")
        ax.set_xlabel(param_name)
        ax.set_ylabel(METRICA_PRINCIPAL)
        ax.set_title("Curva de validación")
        ax.legend()
        fig.savefig(directorio_salida / "validation_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info("Curvas de aprendizaje/validación guardadas en %s", directorio_salida)
    except Exception:
        logger.exception("No fue posible generar las curvas de aprendizaje/validación.")


# =======================================================================
# Persistencia y predicción (reutilizadas por model_deploy.py)
# =======================================================================
def save_model(pipeline: Any, ruta: Path = RUTA_MEJOR_MODELO) -> None:
    """Serializa un pipeline entrenado a disco con `joblib`.

    Args:
        pipeline: Pipeline o estimador ya ajustado.
        ruta: Ruta de destino del archivo `.pkl`.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(pipeline, ruta)
        logger.info("Modelo guardado en %s", ruta)
    except Exception:
        logger.exception("Error al guardar el modelo en %s", ruta)
        raise


def load_model(ruta: Path = RUTA_MEJOR_MODELO) -> Any:
    """Carga un pipeline previamente serializado con `joblib`.

    Args:
        ruta: Ruta del archivo `.pkl` a cargar.

    Returns:
        El pipeline/estimador deserializado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not ruta.exists():
        mensaje = f"No se encontró el modelo en: {ruta}"
        logger.error(mensaje)
        raise FileNotFoundError(mensaje)
    return joblib.load(ruta)


def predict(pipeline: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Genera predicciones de clase y probabilidad para un lote de datos.

    Args:
        pipeline: Pipeline/estimador ya ajustado.
        X: Variables predictoras a evaluar.

    Returns:
        Tupla `(etiquetas_predichas, probabilidades_clase_positiva)`.
    """
    probabilidades = pipeline.predict_proba(X)[:, 1]
    etiquetas = (probabilidades >= 0.5).astype(int)
    return etiquetas, probabilidades


# =======================================================================
# main
# =======================================================================
def main() -> None:
    """Orquesta el entrenamiento, comparación y selección del modelo campeón."""
    try:
        RUTA_ARTEFACTOS.mkdir(parents=True, exist_ok=True)

        dataframe_crudo = load_data()
        X, y, columnas_numericas, columnas_categoricas, estadisticas_preprocesamiento = prepare_dataset(
            dataframe_crudo
        )

        X_train, X_test, y_train, y_test = split_dataset(X, y, ConfiguracionSplit(proporcion_test=0.2))
        X_tr, X_val, y_tr, y_val = split_dataset(
            X_train, y_train, ConfiguracionSplit(proporcion_test=0.2)
        )

        cv, nombre_estrategia_cv = select_validation_strategy()
        cv_optuna = crear_stratified_kfold(N_SPLITS_CV_OPTUNA)

        resultados: dict[str, ResultadoModelo] = {}

        resultados["logistic_regression"] = train_logistic_regression(
            X_tr, y_tr, cv, columnas_numericas, columnas_categoricas
        )
        resultados["random_forest"] = train_random_forest(
            X_tr, y_tr, cv, columnas_numericas, columnas_categoricas
        )
        resultados["xgboost"] = train_xgboost(
            X_tr, y_tr, X_val, y_val, cv_optuna, columnas_numericas, columnas_categoricas
        )
        resultados["lightgbm"] = train_lightgbm(
            X_tr, y_tr, X_val, y_val, cv_optuna, columnas_numericas, columnas_categoricas
        )
        resultados["catboost"] = train_catboost(
            X_tr, y_tr, X_val, y_val, cv_optuna, columnas_numericas, columnas_categoricas
        )

        # Verificación de sesgo del proceso de búsqueda (solo Regresión Logística,
        # ver justificación de costo computacional en el docstring del módulo).
        cv_externo_nested = crear_stratified_kfold(3)
        cv_interno_nested = crear_stratified_kfold(3)
        grid_nested = {"clasificador__C": [0.01, 0.1, 1.0]}
        pipeline_nested = build_model(
            LogisticRegression(
                max_iter=2000, penalty="l2", solver="liblinear",
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            columnas_numericas,
            columnas_categoricas,
        )
        _, media_nested, std_nested = run_nested_cross_validation(
            pipeline_nested, grid_nested, X_tr, y_tr, cv_externo_nested, cv_interno_nested
        )

        # Se usa `pipeline_para_ensamble` (sin early stopping horneado) y no
        # `pipeline`: StackingClassifier clona y reentrena cada base learner
        # internamente, lo que rompe si el estimador exige un `eval_set`.
        estimadores_base_boosting = [
            ("xgboost", resultados["xgboost"].pipeline_para_ensamble),
            ("lightgbm", resultados["lightgbm"].pipeline_para_ensamble),
            ("catboost", resultados["catboost"].pipeline_para_ensamble),
        ]
        resultados["stacking"] = train_stacking(estimadores_base_boosting, X_tr, y_tr, cv)
        resultados["blending"] = train_blending(estimadores_base_boosting, X_val, y_val)

        # --- Selección del campeón: SIEMPRE con la métrica de validación,
        # nunca con X_test, para no sesgar la selección del modelo. ---
        nombre_campeon = max(
            resultados, key=lambda nombre: resultados[nombre].metricas_validacion[f"{METRICA_PRINCIPAL}_cv"]
        )
        campeon = resultados[nombre_campeon]
        logger.info(
            "Modelo campeón: %s (%s_cv=%.4f)",
            nombre_campeon,
            METRICA_PRINCIPAL,
            campeon.metricas_validacion[f"{METRICA_PRINCIPAL}_cv"],
        )

        # --- Leaderboard final: reporte no sesgado sobre X_test, ya con el
        # campeón fijado. ---
        leaderboard: dict[str, Any] = {}
        for nombre, resultado in resultados.items():
            etiquetas_pred, probabilidades_pred = predict(resultado.pipeline, X_test)
            metricas_test = summarize_classification(y_test, etiquetas_pred, probabilidades_pred)
            metricas_test.update(calculate_business_metrics(y_test, probabilidades_pred))
            leaderboard[nombre] = {
                "metricas_test": metricas_test,
                "metricas_validacion": resultado.metricas_validacion,
                "mejores_parametros": resultado.mejores_parametros,
                "tiempo_entrenamiento_segundos": round(resultado.tiempo_entrenamiento_segundos, 2),
                "estrategia_busqueda": resultado.estrategia_busqueda,
            }

        reporte_final = {
            "modelo_campeon": nombre_campeon,
            "estrategia_validacion_cruzada": nombre_estrategia_cv,
            "metrica_principal": METRICA_PRINCIPAL,
            "nested_cv_logistic_regression": {"media": media_nested, "std": std_nested},
            "leaderboard": leaderboard,
        }

        save_model(campeon.pipeline)
        with RUTA_METRICAS.open("w", encoding="utf-8") as archivo_metricas:
            json.dump(reporte_final, archivo_metricas, indent=2, ensure_ascii=False)
        logger.info("Métricas guardadas en %s", RUTA_METRICAS)

        # Se persisten las estadísticas de referencia (medianas de imputación
        # y límites de winsorización) calculadas sobre el set de
        # entrenamiento. `model_deploy.py` las reutiliza tal cual en
        # inferencia para evitar *train/serve skew* (ver docstrings de
        # `ft_engineering.clean_data` / `generate_features`).
        with RUTA_ESTADISTICAS_PREPROCESAMIENTO.open("w", encoding="utf-8") as archivo_stats:
            json.dump(estadisticas_preprocesamiento, archivo_stats, indent=2, ensure_ascii=False)
        logger.info("Estadísticas de preprocesamiento guardadas en %s", RUTA_ESTADISTICAS_PREPROCESAMIENTO)

        # Instantánea de referencia (distribución de entrenamiento) para
        # `model_monitoring.py`: variables ya generadas + target real +
        # probabilidad predicha por el campeón sobre TODO `X_train` (unión
        # de `X_tr` y `X_val`). Es la ventana de comparación contra la que
        # se mide drift de producción.
        etiquetas_train, probabilidades_train = predict(campeon.pipeline, X_train)
        df_referencia = X_train.copy()
        df_referencia[VARIABLE_OBJETIVO] = y_train.values
        df_referencia["prediccion_proba"] = probabilidades_train
        df_referencia.to_csv(RUTA_DATOS_REFERENCIA, index=False)
        logger.info("Datos de referencia para monitoreo guardados en %s", RUTA_DATOS_REFERENCIA)

        # Los ensambles (Stacking/Blending) no exponen importancia nativa.
        # En ese caso se reporta, como referencia interpretativa, la del
        # mejor modelo individual (no ensamble) según la métrica de
        # validación — nunca se inventa una importancia para el ensamble.
        modelo_para_importancia = campeon
        if campeon.nombre in {"stacking", "blending"}:
            nombre_mejor_individual = max(
                (n for n in resultados if n not in {"stacking", "blending"}),
                key=lambda nombre: resultados[nombre].metricas_validacion[f"{METRICA_PRINCIPAL}_cv"],
            )
            modelo_para_importancia = resultados[nombre_mejor_individual]
            logger.info(
                "Campeón '%s' es un ensamble: se reporta importancia de variables "
                "del mejor modelo individual ('%s') como referencia interpretativa.",
                campeon.nombre,
                nombre_mejor_individual,
            )

        tabla_importancia = extract_feature_importance(modelo_para_importancia)
        tabla_importancia.to_csv(RUTA_FEATURE_IMPORTANCE, index=False)
        logger.info("Importancia de variables guardada en %s", RUTA_FEATURE_IMPORTANCE)

        generate_visualizations(campeon, X_test, y_test, directorio_salida=RUTA_GRAFICOS / campeon.nombre)
        if modelo_para_importancia is not campeon:
            generate_visualizations(
                modelo_para_importancia,
                X_test,
                y_test,
                directorio_salida=RUTA_GRAFICOS / modelo_para_importancia.nombre,
            )

        # La curva de validación necesita un hiperparámetro escalar propio del
        # modelo; se usa siempre el mejor modelo INDIVIDUAL (nunca un ensamble,
        # que no tiene un único hiperparámetro representativo).
        nombre_param, rango_param = _obtener_parametro_para_curva_validacion(
            modelo_para_importancia.nombre
        )
        # Igual que en el ensamblado: los pipelines de boosting con early
        # stopping horneado no son clonables sin `eval_set`, por lo que las
        # curvas (que reentrenan el modelo repetidas veces) usan la variante
        # `pipeline_para_ensamble` cuando existe.
        pipeline_para_curvas = modelo_para_importancia.pipeline_para_ensamble or modelo_para_importancia.pipeline
        generate_learning_and_validation_curves(
            pipeline_base=pipeline_para_curvas,
            param_name=nombre_param,
            param_range=rango_param,
            X=X_tr,
            y=y_tr,
            cv=cv,
            directorio_salida=RUTA_GRAFICOS / modelo_para_importancia.nombre,
        )

        logger.info("Entrenamiento del pipeline completado exitosamente.")

    except Exception:
        logger.exception("Fallo no controlado en model_training_evaluation.main()")
        raise


if __name__ == "__main__":
    main()
