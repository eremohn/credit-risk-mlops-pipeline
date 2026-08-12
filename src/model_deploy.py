"""Servicio de inferencia (FastAPI) del pipeline de riesgo crediticio.

Expone el modelo campeón serializado por `model_training_evaluation.py`
a través de una API REST, aceptando solicitudes individuales o por lote
en formato JSON y también carga masiva vía CSV.

Garantía de paridad train/serve:
    Este módulo NO reimplementa limpieza ni ingeniería de variables.
    Reutiliza exactamente `ft_engineering.clean_data` y
    `ft_engineering.generate_features`, alimentadas con las mismas
    medianas de imputación y límites de winsorización calculados sobre
    el dataset de entrenamiento (`preprocessing_stats.json`, generado
    por `model_training_evaluation.py`). Esto evita el error clásico de
    *train/serve skew* en el que la API recalcularía esas estadísticas
    a partir de cada lote entrante.

Contrato de la API:
    El cuerpo de la solicitud replica los campos crudos disponibles en
    el momento de originar un crédito (antes de que existan saldos:
    `saldo_mora`, `saldo_total`, etc. — esas columnas son fuga de
    información y nunca se solicitan ni se usan, ver
    `ft_engineering.COLUMNAS_FUGA_INFORMACION`).

Ejecución local:
    uvicorn model_deploy:app --host 0.0.0.0 --port 8000

La aplicación no depende de rutas absolutas ni de variables de entorno
específicas de un host, por lo que puede empaquetarse en un contenedor
Docker sin cambios (el Dockerfile queda fuera del alcance de este
archivo, según lo solicitado).
"""

from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ft_engineering import clean_data, generate_features
from model_training_evaluation import load_model, predict

# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_deploy")

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------
API_VERSION: Final[str] = "1.0.0"
RUTA_ARTEFACTOS: Final[Path] = Path("../model_artifacts")
RUTA_MODELO: Final[Path] = RUTA_ARTEFACTOS / "best_model.pkl"
RUTA_ESTADISTICAS_PREPROCESAMIENTO: Final[Path] = RUTA_ARTEFACTOS / "preprocessing_stats.json"
RUTA_METRICAS: Final[Path] = RUTA_ARTEFACTOS / "metrics.json"

MAX_SOLICITUDES_POR_LOTE: Final[int] = 5000
UMBRAL_DECISION: Final[float] = 0.5

CATEGORIAS_TIPO_LABORAL_CONOCIDAS: Final[set[str]] = {
    "Empleado",
    "Independiente",
    "Pensionado",
    "Rentista",
}
CATEGORIAS_TENDENCIA_INGRESOS_CONOCIDAS: Final[set[str]] = {
    "Estable",
    "Creciente",
    "Decreciente",
}

COLUMNAS_CSV_REQUERIDAS: Final[list[str]] = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
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
    "tendencia_ingresos",
]

# Columnas que deben quedar en dtype numérico (float) tras construir el
# DataFrame, incluso si vienen con `None`/nulos: cuando un lote JSON de
# una sola solicitud tiene `None` en alguna de estas columnas, pandas
# infiere dtype `object` (Python `None`, no `NaN`) al no tener otro
# valor float en la columna para forzar la inferencia. Sin esta
# conversión explícita, `np.log1p` de `generate_features` falla contra
# ese `object` dtype.
COLUMNAS_NUMERICAS_ENTRADA: Final[list[str]] = [
    "tipo_credito",
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
]


# =======================================================================
# Estado global de la aplicación (modelo y estadísticas cargados al inicio)
# =======================================================================
class EstadoModelo:
    """Contenedor del estado del modelo cargado en memoria.

    Se instancia una única vez al iniciar el servicio (ver `lifespan`) y
    se reutiliza en cada request, evitando recargar el modelo por
    solicitud.

    Attributes:
        pipeline: Pipeline/estimador de scikit-learn ya ajustado.
        medianas_referencia: Medianas de imputación del entrenamiento.
        limites_winsorizacion: Límites de winsorización del entrenamiento.
        nombre_modelo: Nombre del modelo campeón (p. ej. `"stacking"`).
        version_modelo: Marca de versión derivada de la fecha de entrenamiento.
        cargado_correctamente: Indica si la carga fue exitosa.
    """

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.medianas_referencia: dict[str, float] = {}
        self.limites_winsorizacion: dict[str, tuple[float, float]] = {}
        self.nombre_modelo: str = "desconocido"
        self.version_modelo: str = "desconocido"
        self.cargado_correctamente: bool = False


estado_modelo = EstadoModelo()


def cargar_estadisticas_preprocesamiento(
    ruta: Path = RUTA_ESTADISTICAS_PREPROCESAMIENTO,
) -> dict[str, Any]:
    """Carga las estadísticas de referencia (medianas, límites de winsorización).

    Args:
        ruta: Ruta al archivo `preprocessing_stats.json`.

    Returns:
        Diccionario con `medianas_referencia` y `limites_winsorizacion`.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not ruta.exists():
        mensaje = f"No se encontraron las estadísticas de preprocesamiento en: {ruta}"
        logger.error(mensaje)
        raise FileNotFoundError(mensaje)
    with ruta.open("r", encoding="utf-8") as archivo:
        estadisticas = json.load(archivo)
    # JSON no preserva tuplas: se reconvierten explícitamente.
    estadisticas["limites_winsorizacion"] = {
        columna: tuple(limites) for columna, limites in estadisticas["limites_winsorizacion"].items()
    }
    return estadisticas


def cargar_nombre_modelo_campeon(ruta: Path = RUTA_METRICAS) -> str:
    """Obtiene el nombre del modelo campeón desde `metrics.json`.

    Args:
        ruta: Ruta al archivo `metrics.json` generado por el entrenamiento.

    Returns:
        Nombre del modelo campeón, o `"desconocido"` si no puede leerse.
    """
    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            return json.load(archivo).get("modelo_campeon", "desconocido")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("No fue posible leer el nombre del modelo campeón desde %s", ruta)
        return "desconocido"


def inicializar_estado_modelo() -> None:
    """Carga el modelo y sus estadísticas de referencia en `estado_modelo`.

    Se ejecuta una única vez al iniciar la aplicación. Si la carga
    falla, el servicio permanece activo pero reporta `unhealthy` en
    `/health` y rechaza solicitudes de predicción con `503`, en lugar
    de terminar abruptamente el proceso.
    """
    try:
        estado_modelo.pipeline = load_model(RUTA_MODELO)
        estadisticas = cargar_estadisticas_preprocesamiento()
        estado_modelo.medianas_referencia = estadisticas["medianas_referencia"]
        estado_modelo.limites_winsorizacion = estadisticas["limites_winsorizacion"]
        estado_modelo.nombre_modelo = cargar_nombre_modelo_campeon()
        estado_modelo.version_modelo = datetime.fromtimestamp(
            RUTA_MODELO.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y%m%d-%H%M%S")
        estado_modelo.cargado_correctamente = True
        logger.info(
            "Modelo '%s' (versión %s) cargado correctamente.",
            estado_modelo.nombre_modelo,
            estado_modelo.version_modelo,
        )
    except Exception:
        estado_modelo.cargado_correctamente = False
        logger.exception("Fallo al cargar el modelo o sus estadísticas de preprocesamiento.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: carga el modelo al iniciar."""
    inicializar_estado_modelo()
    yield
    logger.info("Servicio de inferencia detenido.")


# =======================================================================
# Esquemas Pydantic
# =======================================================================
class SolicitudCredito(BaseModel):
    """Datos de una solicitud de crédito en el momento de originación.

    Replica exactamente los campos disponibles al momento de evaluar un
    nuevo cliente (antes de que exista historial de pagos del propio
    crédito), consistente con las columnas de fuga de información
    excluidas en `ft_engineering.COLUMNAS_FUGA_INFORMACION`.
    """

    tipo_credito: int = Field(..., ge=1, le=20, description="Código de tipo de crédito.")
    fecha_prestamo: datetime = Field(..., description="Fecha de desembolso propuesta.")
    capital_prestado: float = Field(..., gt=0, description="Monto solicitado.")
    plazo_meses: int = Field(..., gt=0, le=360, description="Plazo del crédito en meses.")
    edad_cliente: int = Field(..., ge=18, le=120, description="Edad del solicitante.")
    tipo_laboral: str = Field(..., description="Situación laboral del solicitante.")
    salario_cliente: float = Field(..., gt=0, description="Salario mensual reportado.")
    total_otros_prestamos: float = Field(..., ge=0, description="Suma de otras deudas vigentes.")
    cuota_pactada: float = Field(..., gt=0, description="Cuota mensual pactada.")
    puntaje: float | None = Field(default=None, description="Score de riesgo interno.")
    puntaje_datacredito: float | None = Field(default=None, description="Score de la central de riesgo.")
    cant_creditosvigentes: int = Field(..., ge=0, description="Créditos vigentes reportados.")
    huella_consulta: int = Field(..., ge=0, description="Consultas recientes a centrales de riesgo.")
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    promedio_ingresos_datacredito: float | None = Field(
        default=None, description="Promedio de ingresos reportado por la central de riesgo."
    )
    tendencia_ingresos: str | None = Field(default=None, description="Tendencia de ingresos reportada.")

    @field_validator("tipo_laboral")
    @classmethod
    def _validar_tipo_laboral(cls, valor: str) -> str:
        if valor not in CATEGORIAS_TIPO_LABORAL_CONOCIDAS:
            logger.warning("tipo_laboral fuera de catálogo conocido: %s", valor)
        return valor

    @field_validator("tendencia_ingresos")
    @classmethod
    def _validar_tendencia_ingresos(cls, valor: str | None) -> str | None:
        if valor is not None and valor not in CATEGORIAS_TENDENCIA_INGRESOS_CONOCIDAS:
            logger.warning("tendencia_ingresos fuera de catálogo conocido: %s", valor)
        return valor

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_credito": 7,
                "fecha_prestamo": "2026-08-08T10:00:00",
                "capital_prestado": 3692160.0,
                "plazo_meses": 10,
                "edad_cliente": 42,
                "tipo_laboral": "Independiente",
                "salario_cliente": 8000000,
                "total_otros_prestamos": 2500000,
                "cuota_pactada": 341296,
                "puntaje": 88.77,
                "puntaje_datacredito": 695,
                "cant_creditosvigentes": 10,
                "huella_consulta": 5,
                "creditos_sectorFinanciero": 5,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 908526,
                "tendencia_ingresos": "Estable",
            }
        }


class LoteSolicitudesCredito(BaseModel):
    """Cuerpo de solicitud del endpoint `/predict`: una o más solicitudes."""

    solicitudes: list[SolicitudCredito] = Field(..., min_length=1, max_length=MAX_SOLICITUDES_POR_LOTE)


class PrediccionCredito(BaseModel):
    """Resultado de la evaluación de riesgo de una solicitud de crédito."""

    prediccion: int = Field(..., description="1 = paga a tiempo, 0 = riesgo de incumplimiento.")
    etiqueta: str
    probabilidad_pago_atiempo: float
    timestamp: str
    modelo_utilizado: str
    version_modelo: str


class RespuestaPrediccionLote(BaseModel):
    """Respuesta estándar de predicción por lote (usada por JSON y CSV)."""

    n_solicitudes: int
    tiempo_procesamiento_ms: float
    predicciones: list[PrediccionCredito]


class RespuestaSalud(BaseModel):
    """Respuesta del endpoint de verificación de salud del servicio."""

    estado: str
    modelo_cargado: bool
    nombre_modelo: str
    version_modelo: str
    version_api: str


# =======================================================================
# Funciones de negocio (reutilizan ft_engineering / model_training_evaluation)
# =======================================================================
def construir_dataframe_desde_solicitudes(solicitudes: list[SolicitudCredito]) -> pd.DataFrame:
    """Convierte una lista de solicitudes Pydantic en un DataFrame crudo.

    Args:
        solicitudes: Solicitudes de crédito ya validadas por Pydantic.

    Returns:
        DataFrame con las mismas columnas y tipos que espera
        `ft_engineering.clean_data` (incluida `fecha_prestamo` como
        tipo fecha y las columnas numéricas forzadas a `float64`, para
        que un `None` aislado en un lote de una sola solicitud no quede
        como dtype `object`).
    """
    dataframe = pd.DataFrame([solicitud.model_dump() for solicitud in solicitudes])
    dataframe["fecha_prestamo"] = pd.to_datetime(dataframe["fecha_prestamo"])
    for columna in COLUMNAS_NUMERICAS_ENTRADA:
        dataframe[columna] = pd.to_numeric(dataframe[columna], errors="coerce")
    return dataframe


def generar_predicciones(dataframe_crudo: pd.DataFrame) -> list[PrediccionCredito]:
    """Ejecuta el pipeline de inferencia completo sobre un lote de solicitudes.

    Reutiliza exactamente las mismas funciones de limpieza e ingeniería
    de variables que el entrenamiento, alimentadas con las estadísticas
    de referencia persistidas (`estado_modelo`), garantizando paridad
    train/serve.

    Args:
        dataframe_crudo: DataFrame con las columnas crudas de una o más
            solicitudes de crédito.

    Returns:
        Lista de `PrediccionCredito`, en el mismo orden que las filas
        de entrada.

    Raises:
        RuntimeError: Si el modelo no fue cargado correctamente.
    """
    if not estado_modelo.cargado_correctamente:
        raise RuntimeError("El modelo no está disponible. Verifique el estado del servicio en /health.")

    df_limpio, _ = clean_data(dataframe_crudo, medianas_referencia=estado_modelo.medianas_referencia)
    df_features, _ = generate_features(df_limpio, limites_winsorizacion=estado_modelo.limites_winsorizacion)

    etiquetas, probabilidades = predict(estado_modelo.pipeline, df_features)
    timestamp_actual = datetime.now(timezone.utc).isoformat()

    return [
        PrediccionCredito(
            prediccion=int(etiqueta),
            etiqueta="pago_a_tiempo" if etiqueta == 1 else "riesgo_incumplimiento",
            probabilidad_pago_atiempo=round(float(probabilidad), 6),
            timestamp=timestamp_actual,
            modelo_utilizado=estado_modelo.nombre_modelo,
            version_modelo=estado_modelo.version_modelo,
        )
        for etiqueta, probabilidad in zip(etiquetas, probabilidades)
    ]


def leer_csv_solicitudes(contenido: bytes) -> pd.DataFrame:
    """Parsea y valida el esquema mínimo de un CSV de solicitudes de crédito.

    Args:
        contenido: Bytes crudos del archivo CSV recibido.

    Returns:
        DataFrame con `fecha_prestamo` parseada como fecha.

    Raises:
        ValueError: Si el CSV está vacío, mal formado, excede el tamaño
            máximo de lote o le faltan columnas requeridas.
    """
    try:
        dataframe = pd.read_csv(io.BytesIO(contenido), parse_dates=["fecha_prestamo"])
    except pd.errors.EmptyDataError as error:
        raise ValueError("El archivo CSV está vacío.") from error
    except (pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"No fue posible parsear el CSV: {error}") from error

    columnas_faltantes = set(COLUMNAS_CSV_REQUERIDAS) - set(dataframe.columns)
    if columnas_faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {sorted(columnas_faltantes)}")

    if len(dataframe) > MAX_SOLICITUDES_POR_LOTE:
        raise ValueError(
            f"El CSV contiene {len(dataframe)} filas; el máximo por lote es {MAX_SOLICITUDES_POR_LOTE}."
        )
    if dataframe.empty:
        raise ValueError("El CSV no contiene filas.")

    for columna in COLUMNAS_NUMERICAS_ENTRADA:
        dataframe[columna] = pd.to_numeric(dataframe[columna], errors="coerce")

    return dataframe


# =======================================================================
# Aplicación FastAPI
# =======================================================================
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Servicio de inferencia del modelo campeón de comportamiento crediticio.",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def obtener_info_servicio() -> dict[str, str]:
    """Retorna información básica del servicio."""
    return {
        "servicio": "API de Riesgo Crediticio",
        "version_api": API_VERSION,
        "documentacion": "/docs",
    }


@app.get("/health", response_model=RespuestaSalud, tags=["General"])
def verificar_salud() -> RespuestaSalud:
    """Reporta el estado del servicio y si el modelo está disponible."""
    return RespuestaSalud(
        estado="ok" if estado_modelo.cargado_correctamente else "unhealthy",
        modelo_cargado=estado_modelo.cargado_correctamente,
        nombre_modelo=estado_modelo.nombre_modelo,
        version_modelo=estado_modelo.version_modelo,
        version_api=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=RespuestaPrediccionLote,
    tags=["Predicción"],
    summary="Predice el riesgo crediticio para una o más solicitudes (JSON).",
)
def predecir_desde_json(lote: LoteSolicitudesCredito) -> RespuestaPrediccionLote:
    """Recibe una o más solicitudes de crédito en JSON y retorna su predicción.

    Una solicitud individual es simplemente un lote de tamaño 1: no hay
    un endpoint separado para el caso single-item, evitando duplicar
    lógica de validación y scoring (DRY).
    """
    if not estado_modelo.cargado_correctamente:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible actualmente. Consulte /health.",
        )
    inicio = time.perf_counter()
    try:
        dataframe_crudo = construir_dataframe_desde_solicitudes(lote.solicitudes)
        predicciones = generar_predicciones(dataframe_crudo)
    except Exception:
        logger.exception("Error al generar predicciones desde JSON.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la solicitud. Verifique los datos de entrada.",
        )
    tiempo_ms = (time.perf_counter() - inicio) * 1000
    logger.info("Predicción JSON: %d solicitud(es) en %.1fms", len(predicciones), tiempo_ms)
    return RespuestaPrediccionLote(
        n_solicitudes=len(predicciones), tiempo_procesamiento_ms=round(tiempo_ms, 2), predicciones=predicciones
    )


@app.post(
    "/predict/csv",
    response_model=RespuestaPrediccionLote,
    tags=["Predicción"],
    summary="Predice el riesgo crediticio para un lote de solicitudes (CSV).",
)
async def predecir_desde_csv(archivo: UploadFile = File(...)) -> RespuestaPrediccionLote:
    """Recibe un archivo CSV con múltiples solicitudes y retorna sus predicciones.

    El CSV debe contener, como mínimo, las columnas listadas en
    `COLUMNAS_CSV_REQUERIDAS` (los mismos campos que `SolicitudCredito`).
    """
    if not estado_modelo.cargado_correctamente:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible actualmente. Consulte /health.",
        )
    if not archivo.filename or not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo debe ser un .csv.")

    inicio = time.perf_counter()
    try:
        contenido = await archivo.read()
        dataframe_crudo = leer_csv_solicitudes(contenido)
        predicciones = generar_predicciones(dataframe_crudo)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    except Exception:
        logger.exception("Error al generar predicciones desde CSV.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar el archivo. Verifique el formato y los datos.",
        )
    tiempo_ms = (time.perf_counter() - inicio) * 1000
    logger.info("Predicción CSV: %d solicitud(es) en %.1fms", len(predicciones), tiempo_ms)
    return RespuestaPrediccionLote(
        n_solicitudes=len(predicciones), tiempo_procesamiento_ms=round(tiempo_ms, 2), predicciones=predicciones
    )


@app.exception_handler(Exception)
async def manejar_error_no_controlado(request, exc: Exception) -> JSONResponse:
    """Red de seguridad final: nunca deja que el proceso termine abruptamente."""
    logger.exception("Error no controlado atendiendo %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Ocurrió un error interno inesperado."},
    )


# =======================================================================
# main
# =======================================================================
def main() -> None:
    """Punto de entrada para ejecución directa (`python model_deploy.py`).

    El modo recomendado en producción sigue siendo lanzar la app vía
    `uvicorn model_deploy:app --host 0.0.0.0 --port 8000`, que es
    además el comando que se invocará desde el contenedor Docker; esta
    función es una conveniencia equivalente para desarrollo local.
    """
    import uvicorn

    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
