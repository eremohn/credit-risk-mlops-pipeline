"""Dashboard de monitoreo del pipeline de riesgo crediticio (Streamlit).

Consume exclusivamente los artefactos ya generados por el resto del
pipeline (`best_model.pkl`, `metrics.json`, `feature_importance.csv`,
`preprocessing_stats.json`, `reference_data.csv`, `drift_report.json`,
`feature_drift.csv`, `monitoring_history.csv`,
`latest_production_snapshot.csv`). El dashboard nunca reentrena ni
recalcula drift: es una capa de visualización sobre lo que
`model_training_evaluation.py` y `model_monitoring.py` ya calcularon y
persistieron, consistente con el resto del proyecto.

La única pieza que ejecuta inferencia en vivo es el simulador de
crédito (pestaña "Simulador"), que reutiliza `ft_engineering.clean_data`
/ `generate_features` y `model_training_evaluation.load_model` /
`predict` — el mismo camino de código que `model_deploy.py` — para
garantizar que el resultado mostrado en el dashboard sea idéntico al
que devolvería la API.

Ejecución:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ft_engineering import (
    VARIABLE_OBJETIVO,
    clean_data,
    generate_features,
)
from model_training_evaluation import load_model, predict

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
RUTA_ARTEFACTOS: Final[Path] = Path("../model_artifacts")
RUTA_MODELO: Final[Path] = RUTA_ARTEFACTOS / "best_model.pkl"
RUTA_METRICAS: Final[Path] = RUTA_ARTEFACTOS / "metrics.json"
RUTA_FEATURE_IMPORTANCE: Final[Path] = RUTA_ARTEFACTOS / "feature_importance.csv"
RUTA_ESTADISTICAS_PREPROCESAMIENTO: Final[Path] = RUTA_ARTEFACTOS / "preprocessing_stats.json"
RUTA_DATOS_REFERENCIA: Final[Path] = RUTA_ARTEFACTOS / "reference_data.csv"
RUTA_REPORTE_DRIFT: Final[Path] = RUTA_ARTEFACTOS / "drift_report.json"
RUTA_TABLA_DRIFT: Final[Path] = RUTA_ARTEFACTOS / "feature_drift.csv"
RUTA_HISTORICO_MONITOREO: Final[Path] = RUTA_ARTEFACTOS / "monitoring_history.csv"
RUTA_SNAPSHOT_PRODUCCION: Final[Path] = RUTA_ARTEFACTOS / "latest_production_snapshot.csv"

UMBRAL_SEMAFORO_VERDE: Final[float] = 0.80
UMBRAL_SEMAFORO_AMARILLO: Final[float] = 0.50

COLUMNAS_NO_PREDICTORAS: Final[list[str]] = [VARIABLE_OBJETIVO, "prediccion_proba"]

# Paleta: terminal financiero oscuro. El color solo se usa con carga
# semántica (verde/ámbar/rojo = riesgo); el resto de la UI se mantiene
# monocromática a propósito para que el semáforo sea lo único que
# "grite" en la pantalla.
PALETA: Final[dict[str, str]] = {
    "fondo": "#0B1220",
    "tarjeta": "#131B2E",
    "borde": "#232C42",
    "texto": "#E7EAF2",
    "texto_tenue": "#8892A6",
    "acento": "#3E7CB1",
    "verde": "#2E8B57",
    "amarillo": "#D4A017",
    "rojo": "#C0392B",
}


# =======================================================================
# Estilos
# =======================================================================
def inyectar_estilos() -> None:
    """Inyecta la hoja de estilos del dashboard (tipografía + tarjetas + semáforo)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {PALETA['fondo']};
            color: {PALETA['texto']};
        }}
        .tarjeta-kpi {{
            background-color: {PALETA['tarjeta']};
            border: 1px solid {PALETA['borde']};
            border-radius: 8px;
            padding: 16px 20px;
        }}
        .tarjeta-kpi .etiqueta {{
            font-size: 0.75rem;
            color: {PALETA['texto_tenue']};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .tarjeta-kpi .valor {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: {PALETA['texto']};
        }}
        .semaforo {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.95rem;
        }}
        .semaforo-verde {{ background-color: {PALETA['verde']}22; color: {PALETA['verde']}; border: 1px solid {PALETA['verde']}; }}
        .semaforo-amarillo {{ background-color: {PALETA['amarillo']}22; color: {PALETA['amarillo']}; border: 1px solid {PALETA['amarillo']}; }}
        .semaforo-rojo {{ background-color: {PALETA['rojo']}22; color: {PALETA['rojo']}; border: 1px solid {PALETA['rojo']}; }}
        .alerta {{
            border-left: 4px solid {PALETA['rojo']};
            background-color: {PALETA['rojo']}15;
            padding: 10px 14px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }}
        .alerta-moderada {{
            border-left: 4px solid {PALETA['amarillo']};
            background-color: {PALETA['amarillo']}15;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _configurar_matplotlib_oscuro() -> None:
    """Configura matplotlib para que los gráficos combinen con el tema oscuro."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETA["fondo"],
            "axes.facecolor": PALETA["tarjeta"],
            "axes.edgecolor": PALETA["borde"],
            "axes.labelcolor": PALETA["texto"],
            "text.color": PALETA["texto"],
            "xtick.color": PALETA["texto_tenue"],
            "ytick.color": PALETA["texto_tenue"],
            "grid.color": PALETA["borde"],
            "font.family": "sans-serif",
        }
    )


# =======================================================================
# Carga de artefactos (cacheada: el dashboard solo lee, nunca recalcula)
# =======================================================================
@st.cache_data(show_spinner=False)
def cargar_json(ruta: Path) -> dict[str, Any] | None:
    """Carga un archivo JSON de artefactos, o `None` si no existe."""
    if not ruta.exists():
        return None
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_data(show_spinner=False)
def cargar_csv(ruta: Path) -> pd.DataFrame | None:
    """Carga un CSV de artefactos, o `None` si no existe."""
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


@st.cache_resource(show_spinner="Cargando modelo campeón...")
def cargar_modelo_cacheado() -> Any:
    """Carga el pipeline del modelo campeón (una sola vez por sesión)."""
    return load_model(RUTA_MODELO)


def cargar_estadisticas_preprocesamiento() -> dict[str, Any] | None:
    """Carga medianas de imputación y límites de winsorización de referencia.

    Se implementa localmente (no se importa desde `model_deploy.py`)
    para que el dashboard no dependa de FastAPI, una librería propia
    del servicio de inferencia y ajena a las necesidades de una app de
    Streamlit.
    """
    estadisticas = cargar_json(RUTA_ESTADISTICAS_PREPROCESAMIENTO)
    if estadisticas is None:
        return None
    estadisticas = dict(estadisticas)
    estadisticas["limites_winsorizacion"] = {
        columna: tuple(limites) for columna, limites in estadisticas["limites_winsorizacion"].items()
    }
    return estadisticas


# =======================================================================
# Lógica de negocio: semáforo de riesgo
# =======================================================================
def construir_semaforo_credito(probabilidad_pago_atiempo: float) -> tuple[str, str]:
    """Clasifica una probabilidad de pago a tiempo en un semáforo de riesgo.

    Args:
        probabilidad_pago_atiempo: Probabilidad predicha por el modelo.

    Returns:
        Tupla `(color, etiqueta)`, con `color` en `{"verde","amarillo","rojo"}`.
    """
    if probabilidad_pago_atiempo >= UMBRAL_SEMAFORO_VERDE:
        return "verde", "Riesgo bajo — aprobación recomendada"
    if probabilidad_pago_atiempo >= UMBRAL_SEMAFORO_AMARILLO:
        return "amarillo", "Riesgo medio — revisión manual sugerida"
    return "rojo", "Riesgo alto — no recomendado"


def construir_semaforo_drift(resumen_ejecutivo: dict[str, Any]) -> tuple[str, str]:
    """Clasifica el estado global de drift del sistema en un semáforo.

    Args:
        resumen_ejecutivo: Bloque `resumen_ejecutivo` de `drift_report.json`.

    Returns:
        Tupla `(color, etiqueta)`.
    """
    severidades = {
        resumen_ejecutivo["drift_prediccion"],
        resumen_ejecutivo["drift_target"],
    }
    if resumen_ejecutivo["variables_con_drift_severo"] or "drift_severo" in severidades:
        return "rojo", "Drift severo detectado — revisar reentrenamiento"
    if resumen_ejecutivo["variables_con_drift_moderado"] or "drift_moderado" in severidades:
        return "amarillo", "Drift moderado — monitorear de cerca"
    return "verde", "Sin drift relevante"


def renderizar_semaforo(color: str, etiqueta: str) -> None:
    """Renderiza una insignia de semáforo con la etiqueta dada."""
    st.markdown(f'<span class="semaforo semaforo-{color}">● {etiqueta}</span>', unsafe_allow_html=True)


# =======================================================================
# Componentes de UI reutilizables
# =======================================================================
def renderizar_tarjeta_kpi(columna, etiqueta: str, valor: str) -> None:
    """Renderiza una tarjeta KPI individual dentro de una columna de Streamlit."""
    columna.markdown(
        f"""<div class="tarjeta-kpi"><div class="etiqueta">{etiqueta}</div>
        <div class="valor">{valor}</div></div>""",
        unsafe_allow_html=True,
    )


def renderizar_kpis_modelo(metricas: dict[str, Any]) -> None:
    """Renderiza la fila de KPIs del modelo campeón.

    Args:
        metricas: Contenido completo de `metrics.json`.
    """
    nombre_campeon = metricas["modelo_campeon"]
    metricas_test = metricas["leaderboard"][nombre_campeon]["metricas_test"]

    columnas = st.columns(5)
    renderizar_tarjeta_kpi(columnas[0], "Modelo campeón", nombre_campeon.replace("_", " ").title())
    renderizar_tarjeta_kpi(columnas[1], "PR AUC", f"{metricas_test['pr_auc']:.4f}")
    renderizar_tarjeta_kpi(columnas[2], "ROC AUC", f"{metricas_test['roc_auc']:.4f}")
    renderizar_tarjeta_kpi(columnas[3], "KS Statistic", f"{metricas_test['ks_statistic']:.4f}")
    renderizar_tarjeta_kpi(columnas[4], "Recall", f"{metricas_test['recall']:.4f}")


def renderizar_alertas(resumen_ejecutivo: dict[str, Any]) -> None:
    """Renderiza banners de alerta según la severidad de drift detectada.

    Args:
        resumen_ejecutivo: Bloque `resumen_ejecutivo` de `drift_report.json`.
    """
    if resumen_ejecutivo["variables_con_drift_severo"]:
        variables = ", ".join(resumen_ejecutivo["variables_con_drift_severo"])
        st.markdown(f'<div class="alerta">⚠ Drift severo en: <b>{variables}</b></div>', unsafe_allow_html=True)
    if resumen_ejecutivo["variables_con_drift_moderado"]:
        variables = ", ".join(resumen_ejecutivo["variables_con_drift_moderado"])
        st.markdown(
            f'<div class="alerta alerta-moderada">⚠ Drift moderado en: <b>{variables}</b></div>',
            unsafe_allow_html=True,
        )
    if resumen_ejecutivo["drift_prediccion"] == "drift_severo":
        st.markdown(
            '<div class="alerta">⚠ La distribución de <b>probabilidades predichas</b> '
            "se desvió significativamente de entrenamiento.</div>",
            unsafe_allow_html=True,
        )
    if resumen_ejecutivo["drift_target"] == "drift_severo":
        st.markdown(
            '<div class="alerta">⚠ La <b>tasa real de incumplimiento</b> en producción '
            "se desvió significativamente de entrenamiento.</div>",
            unsafe_allow_html=True,
        )
    if not any(
        [
            resumen_ejecutivo["variables_con_drift_severo"],
            resumen_ejecutivo["variables_con_drift_moderado"],
            resumen_ejecutivo["drift_prediccion"] != "sin_drift",
            resumen_ejecutivo["drift_target"] not in {"sin_drift", "no_disponible"},
        ]
    ):
        st.success("Sin alertas activas. El sistema opera dentro de los rangos esperados.")


# =======================================================================
# Pestañas
# =======================================================================
def pestana_resumen(metricas: dict[str, Any] | None, reporte_drift: dict[str, Any] | None) -> None:
    """Pestaña de resumen ejecutivo: KPIs, semáforo global y alertas."""
    st.subheader("Resumen ejecutivo")

    if metricas is None:
        st.warning("No se encontró `metrics.json`. Ejecute `model_training_evaluation.py` primero.")
        return
    renderizar_kpis_modelo(metricas)
    st.markdown("")

    if reporte_drift is None:
        st.info("Aún no hay corridas de monitoreo. Ejecute `model_monitoring.py` para ver el estado de drift.")
        return

    col_semaforo, col_alertas = st.columns([1, 2])
    with col_semaforo:
        st.markdown("**Estado del sistema**")
        color, etiqueta = construir_semaforo_drift(reporte_drift["resumen_ejecutivo"])
        renderizar_semaforo(color, etiqueta)
        st.caption(f"Referencia: {reporte_drift['n_referencia']} créditos · Producción: {reporte_drift['n_produccion']} créditos")
    with col_alertas:
        st.markdown("**Alertas**")
        renderizar_alertas(reporte_drift["resumen_ejecutivo"])


def pestana_desempeno(metricas: dict[str, Any] | None, df_importancia: pd.DataFrame | None) -> None:
    """Pestaña de desempeño del modelo: leaderboard y variables más influyentes."""
    st.subheader("Desempeño del modelo")

    if metricas is None:
        st.warning("No se encontró `metrics.json`. Ejecute `model_training_evaluation.py` primero.")
        return

    st.markdown("**Tabla comparativa de modelos (leaderboard)**")
    filas = []
    for nombre_modelo, resultado in metricas["leaderboard"].items():
        fila = {"modelo": nombre_modelo, "es_campeon": nombre_modelo == metricas["modelo_campeon"]}
        fila.update(resultado["metricas_test"])
        fila["tiempo_entrenamiento_segundos"] = resultado["tiempo_entrenamiento_segundos"]
        filas.append(fila)
    tabla_leaderboard = pd.DataFrame(filas).sort_values("pr_auc", ascending=False)
    st.dataframe(tabla_leaderboard, width='stretch', hide_index=True)

    if df_importancia is not None and not df_importancia.empty:
        st.markdown(f"**Variables más influyentes** (modelo: `{df_importancia['modelo'].iloc[0]}`)")
        top_variables = df_importancia.head(15).sort_values("importancia")
        _configurar_matplotlib_oscuro()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_variables["variable"], top_variables["importancia"], color=PALETA["acento"])
        ax.set_xlabel("Importancia")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No hay importancia de variables disponible (el campeón fue un ensamble sin referencia individual guardada).")


def pestana_drift(
    reporte_drift: dict[str, Any] | None, tabla_drift: pd.DataFrame | None, historico: pd.DataFrame | None
) -> None:
    """Pestaña de monitoreo de drift: tabla, PSI por variable y tendencia temporal."""
    st.subheader("Monitoreo de drift")

    if reporte_drift is None or tabla_drift is None:
        st.warning("No hay reporte de drift disponible. Ejecute `model_monitoring.py` primero.")
        return

    renderizar_alertas(reporte_drift["resumen_ejecutivo"])
    st.markdown("**PSI por variable** (referencia vs. producción)")

    _configurar_matplotlib_oscuro()
    tabla_ordenada = tabla_drift.sort_values("psi", ascending=True)
    colores_barras = tabla_ordenada["severidad"].map(
        {"sin_drift": PALETA["verde"], "drift_moderado": PALETA["amarillo"], "drift_severo": PALETA["rojo"]}
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(tabla_ordenada) * 0.3)))
    ax.barh(tabla_ordenada["variable"], tabla_ordenada["psi"], color=colores_barras)
    ax.axvline(0.10, color=PALETA["texto_tenue"], linestyle="--", linewidth=1, label="Umbral moderado (0.10)")
    ax.axvline(0.25, color=PALETA["texto_tenue"], linestyle=":", linewidth=1, label="Umbral severo (0.25)")
    ax.set_xlabel("PSI")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Tabla de drift por variable**")
    st.dataframe(
        tabla_drift.sort_values("psi", ascending=False),
        width='stretch',
        hide_index=True,
    )

    st.markdown("**Drift temporal** (histórico de corridas de monitoreo)")
    if historico is None or len(historico) < 2:
        st.info("Se necesitan al menos 2 corridas de `model_monitoring.py` para graficar la tendencia.")
    else:
        historico_indexado = historico.copy()
        historico_indexado["timestamp"] = pd.to_datetime(historico_indexado["timestamp"])
        historico_indexado = historico_indexado.set_index("timestamp")
        st.line_chart(
            historico_indexado[["drift_prediccion_psi", "n_variables_drift_severo", "n_variables_drift_moderado"]]
        )


def pestana_distribuciones(df_referencia: pd.DataFrame | None, df_produccion: pd.DataFrame | None) -> None:
    """Pestaña de distribuciones: comparación referencia vs. producción por variable."""
    st.subheader("Distribuciones: referencia vs. producción")

    if df_referencia is None:
        st.warning("No se encontraron datos de referencia. Ejecute `model_training_evaluation.py` primero.")
        return
    if df_produccion is None:
        st.info("No hay snapshot de producción todavía. Ejecute `model_monitoring.py` primero.")
        return

    columnas_numericas = [
        columna
        for columna in df_referencia.columns
        if columna not in COLUMNAS_NO_PREDICTORAS and pd.api.types.is_numeric_dtype(df_referencia[columna])
    ]
    variable_seleccionada = st.selectbox("Variable", columnas_numericas, index=0)

    _configurar_matplotlib_oscuro()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        df_referencia[variable_seleccionada].dropna(), bins=30, alpha=0.55, density=True,
        label="Referencia (entrenamiento)", color=PALETA["acento"],
    )
    ax.hist(
        df_produccion[variable_seleccionada].dropna(), bins=30, alpha=0.55, density=True,
        label="Producción", color=PALETA["amarillo"],
    )
    ax.set_xlabel(variable_seleccionada)
    ax.set_ylabel("Densidad")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Distribución de la probabilidad predicha**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(
        df_referencia["prediccion_proba"].dropna(), bins=30, alpha=0.55, density=True,
        label="Referencia", color=PALETA["acento"],
    )
    ax2.hist(
        df_produccion["prediccion_proba"].dropna(), bins=30, alpha=0.55, density=True,
        label="Producción", color=PALETA["amarillo"],
    )
    ax2.set_xlabel("Probabilidad de pago a tiempo")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)


def pestana_simulador(estadisticas_preprocesamiento: dict[str, Any] | None) -> None:
    """Pestaña de simulador: evalúa una solicitud de crédito hipotética en vivo."""
    st.subheader("Simulador de solicitud de crédito")

    if estadisticas_preprocesamiento is None:
        st.warning("No se encontraron estadísticas de preprocesamiento. Ejecute el entrenamiento primero.")
        return

    with st.form("formulario_simulador"):
        col1, col2, col3 = st.columns(3)
        with col1:
            capital_prestado = st.number_input("Capital solicitado", min_value=1.0, value=3_000_000.0, step=100_000.0)
            plazo_meses = st.number_input("Plazo (meses)", min_value=1, max_value=360, value=12)
            edad_cliente = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35)
            tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente", "Pensionado", "Rentista"])
        with col2:
            salario_cliente = st.number_input("Salario mensual", min_value=1.0, value=3_000_000.0, step=100_000.0)
            total_otros_prestamos = st.number_input("Otras deudas vigentes", min_value=0.0, value=500_000.0, step=50_000.0)
            cuota_pactada = st.number_input("Cuota mensual pactada", min_value=1.0, value=250_000.0, step=10_000.0)
            tendencia_ingresos = st.selectbox("Tendencia de ingresos", ["Estable", "Creciente", "Decreciente"])
        with col3:
            puntaje_datacredito = st.number_input("Score central de riesgo", min_value=0.0, max_value=999.0, value=750.0)
            huella_consulta = st.number_input("Consultas recientes", min_value=0, value=2)
            creditos_sectorFinanciero = st.number_input("Créditos sector financiero", min_value=0, value=1)
            promedio_ingresos_datacredito = st.number_input("Promedio ingresos reportado", min_value=0.0, value=3_000_000.0, step=100_000.0)

        enviado = st.form_submit_button("Evaluar solicitud")

    if not enviado:
        return

    solicitud_cruda = pd.DataFrame(
        [
            {
                "tipo_credito": 1,
                "fecha_prestamo": pd.Timestamp.now(),
                "capital_prestado": capital_prestado,
                "plazo_meses": plazo_meses,
                "edad_cliente": edad_cliente,
                "tipo_laboral": tipo_laboral,
                "salario_cliente": salario_cliente,
                "total_otros_prestamos": total_otros_prestamos,
                "cuota_pactada": cuota_pactada,
                "puntaje": puntaje_datacredito,
                "puntaje_datacredito": puntaje_datacredito,
                "cant_creditosvigentes": creditos_sectorFinanciero,
                "huella_consulta": huella_consulta,
                "creditos_sectorFinanciero": creditos_sectorFinanciero,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": promedio_ingresos_datacredito,
                "tendencia_ingresos": tendencia_ingresos,
            }
        ]
    )

    try:
        pipeline = cargar_modelo_cacheado()
        df_limpio, _ = clean_data(
            solicitud_cruda, medianas_referencia=estadisticas_preprocesamiento["medianas_referencia"]
        )
        df_features, _ = generate_features(
            df_limpio, limites_winsorizacion=estadisticas_preprocesamiento["limites_winsorizacion"]
        )
        _, probabilidades = predict(pipeline, df_features)
        probabilidad = float(probabilidades[0])
    except Exception:
        logger.exception("Error al evaluar la solicitud del simulador.")
        st.error("No fue posible evaluar la solicitud. Revise los datos ingresados.")
        return

    color, etiqueta = construir_semaforo_credito(probabilidad)
    st.markdown("### Resultado")
    renderizar_semaforo(color, etiqueta)
    st.metric("Probabilidad de pago a tiempo", f"{probabilidad:.1%}")


# =======================================================================
# main
# =======================================================================
def main() -> None:
    """Configura la página y orquesta el renderizado de las pestañas."""
    st.set_page_config(page_title="Riesgo Crediticio · MLOps", page_icon="📊", layout="wide")
    inyectar_estilos()

    st.title("Riesgo Crediticio")
    st.caption("Monitoreo del pipeline de comportamiento crediticio")

    metricas = cargar_json(RUTA_METRICAS)
    df_importancia = cargar_csv(RUTA_FEATURE_IMPORTANCE)
    df_referencia = cargar_csv(RUTA_DATOS_REFERENCIA)
    reporte_drift = cargar_json(RUTA_REPORTE_DRIFT)
    tabla_drift = cargar_csv(RUTA_TABLA_DRIFT)
    historico = cargar_csv(RUTA_HISTORICO_MONITOREO)
    df_produccion = cargar_csv(RUTA_SNAPSHOT_PRODUCCION)
    estadisticas_preprocesamiento = cargar_estadisticas_preprocesamiento()

    tab_resumen, tab_desempeno, tab_drift, tab_distribuciones, tab_simulador = st.tabs(
        ["Resumen", "Desempeño", "Drift", "Distribuciones", "Simulador"]
    )
    with tab_resumen:
        pestana_resumen(metricas, reporte_drift)
    with tab_desempeno:
        pestana_desempeno(metricas, df_importancia)
    with tab_drift:
        pestana_drift(reporte_drift, tabla_drift, historico)
    with tab_distribuciones:
        pestana_distribuciones(df_referencia, df_produccion)
    with tab_simulador:
        pestana_simulador(estadisticas_preprocesamiento)


if __name__ == "__main__":
    main()
