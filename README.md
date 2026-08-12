# 🎓 Credit Risk MLOps Pipeline — Rama `certification`

![Branch](https://img.shields.io/badge/branch-certification-8A2BE2)
![Status](https://img.shields.io/badge/status-entregable%20congelado-success)

Esta rama es el **entregable congelado** del proyecto integrador para efectos de certificación/evaluación académica. A diferencia de [`main`](../../tree/main) (documentación del producto) y [`developer`](../../tree/developer) (guía de contribución), este README está dirigido a un **evaluador** que necesita verificar, en el menor tiempo posible, qué se implementó y cómo comprobarlo.

> 🔒 Esta rama no recibe nuevos merges tras el corte de entrega. Cualquier corrección posterior se realiza sobre `developer`/`main` y, si corresponde, se vuelve a congelar en una nueva rama de certificación.

---

## 📑 Índice

- [Objetivo del proyecto integrador](#-objetivo-del-proyecto-integrador)
- [Contexto y alcance](#-contexto-y-alcance)
- [Cómo evaluar este proyecto](#-cómo-evaluar-este-proyecto)
- [Matriz de trazabilidad: requisito → evidencia](#-matriz-de-trazabilidad-requisito--evidencia)
- [Evidencia de ejecución real](#-evidencia-de-ejecución-real)
- [Decisiones técnicas defendibles](#-decisiones-técnicas-defendibles)
- [Limitaciones declaradas](#-limitaciones-declaradas)
- [Autor y declaración de autenticidad](#-autor-y-declaración-de-autenticidad)

---

## 🎯 Objetivo del proyecto integrador

Construir, de forma individual, un pipeline de Machine Learning **completo y productivo** (no un notebook de experimentación) para predecir el comportamiento crediticio de nuevos clientes de una empresa financiera ficticia, cubriendo: comprensión del negocio, EDA, ingeniería de variables, entrenamiento comparativo de múltiples familias de modelos, optimización de hiperparámetros, ensamblado, selección automática del mejor modelo, despliegue como servicio REST, monitoreo de drift y visualización operativa — aplicando estándares de ingeniería de software (PEP 8/257, tipado, logging, manejo de errores, modularidad) equivalentes a los usados en equipos de MLOps de empresas financieras reales.

## 🏢 Contexto y alcance

Dataset histórico de **10.763 créditos** (`base_de_datos.csv`, 23 columnas), con fuerte desbalance de clases (~95.3 % de créditos pagados a tiempo). El alcance cubierto es el ciclo de vida completo del modelo, desde el dato crudo hasta un dashboard de monitoreo — **no** incluye la infraestructura de despliegue en sí (Docker/CI-CD quedan como preparación, no como implementación; ver [limitaciones declaradas](#-limitaciones-declaradas)).

## 🧭 Cómo evaluar este proyecto

```bash
git clone https://github.com/eremohn/credit-risk-mlops-pipeline.git
cd credit-risk-mlops-pipeline
git checkout certification

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src

# 1) Entrenamiento completo (genera todos los artefactos evaluables)
python model_training_evaluation.py

# 2) Servicio de inferencia
uvicorn model_deploy:app --port 8000 &
curl http://localhost:8000/health

# 3) Monitoreo de drift
python model_monitoring.py

# 4) Dashboard
streamlit run dashboard.py
```

Cada paso produce artefactos verificables (JSON/CSV/PNG) en `model_artifacts/`, listados en la [matriz de trazabilidad](#-matriz-de-trazabilidad-requisito--evidencia).

## 🗂️ Matriz de trazabilidad: requisito → evidencia

| Requisito del proyecto integrador | Implementado en | Evidencia verificable |
|---|---|---|
| Carga y comprensión de datos | `cargar_datos.ipynb`, `comprension_eda.ipynb` | Notebooks del repositorio |
| Limpieza reproducible | `ft_engineering.py::clean_data` | Corrección de puntajes negativos, categorías espurias, edades imposibles |
| Ingeniería de variables | `ft_engineering.py::generate_features` | `ratio_cuota_salario`, `mes_prestamo`, `trimestre_prestamo`, `total_creditos_sector`, `categoria_riesgo_score`, winsorización + `log1p` |
| Selección de variables | `ft_engineering.py::select_features` | Descarte automático por correlación (> 0.9) |
| Regresión Logística regularizada | `model_training_evaluation.py::train_logistic_regression` | `GRID_LOGISTIC_REGRESSION`, `GridSearchCV` |
| Random Forest optimizado | `model_training_evaluation.py::train_random_forest` | `rf_comparacion_busqueda.csv` (Grid vs. Random Search) |
| XGBoost completo (early stopping, gain importance) | `model_training_evaluation.py::train_xgboost` | `entrenar_xgboost_con_early_stopping` |
| LightGBM completo | `model_training_evaluation.py::train_lightgbm` | `entrenar_lightgbm_con_early_stopping` |
| CatBoost con categóricas nativas | `model_training_evaluation.py::train_catboost`, `CatBoostWrapperClassifier` | `create_preprocessing_pipeline_catboost` (sin one-hot) |
| `StratifiedKFold` / `GroupKFold` / `StratifiedGroupKFold` / `TimeSeriesSplit` | `model_training_evaluation.py::select_validation_strategy` | Las 4 estrategias implementadas; selección automática documentada |
| `GridSearchCV` / `RandomizedSearchCV` / Optuna / Nested CV | `model_training_evaluation.py` | `run_grid_search`, `run_random_search`, `run_optuna_search`, `run_nested_cross_validation` |
| Stacking | `model_training_evaluation.py::build_stacking_ensemble` | `StackingClassifier` con meta-learner de Regresión Logística |
| Blending | `model_training_evaluation.py::BlendingClassifier` | Optimización de pesos vía `scipy.optimize.minimize` sobre holdout |
| Métricas (incl. KS, Lift, Gain) | `ft_engineering.py::summarize_classification`, `model_training_evaluation.py::calculate_lift_gain` | `metrics.json` |
| Selección automática del mejor modelo | `model_training_evaluation.py::main` | `metrics.json["modelo_campeon"]`, selección por PR AUC de validación |
| API REST con FastAPI | `model_deploy.py` | Endpoints `/predict`, `/predict/csv`, `/health` — `/docs` interactivo |
| Monitoreo: KS, PSI, Jensen-Shannon, Chi² | `model_monitoring.py` | `calculate_ks_test`, `calculate_psi`, `calculate_jensen_shannon`, `calculate_chi_square` |
| Feature / Prediction / Target drift | `model_monitoring.py::detect_feature_drift/detect_prediction_drift/detect_target_drift` | `drift_report.json` |
| Dashboard Streamlit (semáforo, KPIs, históricos, alertas, gráficos, tablas, drift temporal, distribuciones) | `dashboard.py` | 5 pestañas: Resumen, Desempeño, Drift, Distribuciones, Simulador |

## 🔬 Evidencia de ejecución real

Todos los módulos fueron **ejecutados de punta a punta contra el dataset real** durante el desarrollo (no solo revisados estáticamente). Resultado de una corrida completa de `model_training_evaluation.py`:

| Modelo | PR AUC (test) | ROC AUC | KS |
|---|---|---|---|
| Regresión Logística | 0.9953 | 0.9361 | 0.7607 |
| Random Forest | 0.9962 | 0.9443 | 0.7648 |
| XGBoost | 0.9955 | 0.9352 | 0.7697 |
| LightGBM | 0.9961 | 0.9376 | 0.7557 |
| CatBoost | 0.9962 | 0.9421 | 0.7679 |
| Blending | 0.9964 | 0.9422 | 0.7790 |
| **Stacking (campeón)** | **0.9964** | 0.9421 | 0.7780 |

Ejecutar `python model_training_evaluation.py` reproduce esta tabla (con variación menor por la naturaleza estocástica de Optuna) en `model_artifacts/metrics.json`.

## 🧠 Decisiones técnicas defendibles

Estas son decisiones de diseño que un evaluador puede cuestionar en una defensa oral, junto con su justificación:

1. **¿Por qué PR AUC y no accuracy?** El target tiene ~95 % de casos positivos; un modelo trivial que siempre prediga "paga a tiempo" ya tendría 95 % de accuracy sin haber aprendido nada. PR AUC penaliza correctamente el desempeño sobre la clase minoritaria.
2. **¿Por qué la selección del campeón usa validación y no el conjunto de test?** Para que la métrica de test reportada en `metrics.json` sea una estimación no sesgada del desempeño real, y no un número optimizado indirectamente al elegir el modelo que mejor le va justo en esos datos.
3. **¿Por qué CatBoost tiene un preprocesador distinto al resto?** Para aprovechar su manejo nativo de variables categóricas (`cat_features`), su principal diferenciador frente a los demás modelos — usar one-hot encoding también en CatBoost habría anulado esa ventaja.
4. **¿Por qué `model_deploy.py` y `model_monitoring.py` no recalculan medianas/percentiles de sus propios datos de entrada?** Porque hacerlo introduciría *train/serve skew*: una solicitud de un solo crédito no puede definir "la mediana" de una variable. Esas estadísticas se calculan una única vez en entrenamiento y se persisten en `preprocessing_stats.json`.
5. **¿Por qué no se implementó SHAP?** Fuera de la lista de librerías aprobadas para este proyecto; se usa importancia por ganancia como alternativa dentro del stack permitido.

## ⚠️ Limitaciones declaradas

En cumplimiento del principio de no sobre-representar el alcance del proyecto:

- El **lote de "producción"** que usa `model_monitoring.py` por defecto es una simulación construida a partir de `X_test` (datos históricos nunca usados en entrenamiento), documentada explícitamente en el código — el proyecto no dispone de tráfico productivo real.
- **No se incluye `Dockerfile`** ni un pipeline de CI/CD funcional: el código está preparado para ambos (sin rutas absolutas, entrypoint estándar vía `uvicorn`, logging a stdout), pero su implementación quedó fuera del alcance de esta entrega.
- **No hay suite de pruebas unitarias automatizadas** (`pytest`); la validación de cada módulo se realizó ejecutándolo íntegramente contra los datos reales del proyecto.
- Los hiperparámetros de búsqueda (`N_TRIALS_OPTUNA=15`, entre otros) están fijados a valores moderados para que el pipeline completo pueda ejecutarse y verificarse en minutos; están documentados como ajustables para un entorno de entrenamiento productivo con mayor presupuesto de cómputo.

## 👤 Autor y declaración de autenticidad

Proyecto desarrollado de forma individual como entregable de certificación. La empresa financiera y el dataset descritos son ficticios, usados exclusivamente con fines educativos.

**Rol:** Junior Advanced Data Scientist
**Repositorio:** [github.com/eremohn/credit-risk-mlops-pipeline](https://github.com/eremohn/credit-risk-mlops-pipeline) — rama `certification`
