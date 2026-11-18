"""FastAPI routes and the composition root (Section 9.6).

This module is the one place in the repository allowed to name concrete
classes from several layers at once. A composition root has to: it is what
wires the abstractions together. Everywhere else, a module depends on the
interfaces of ``ai/interfaces.py`` and ``persistence/interfaces.py``, and the
import-direction test asserts that this file is the only exception.

The composition reproduces the snippet of Section 9.6 literally:

    base      = ModelFactory.create(settings.ACTIVE_STRATEGY)   # registry lookup
    decorated = LoggingDecorator(TimingDecorator(CachingDecorator(base)))
    controller = RecommendationController(strategy=decorated, ...)

It runs once, at startup, so the request path performs no wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai.decorators import build_chain
from ai.decorators.caching_decorator import CachingDecorator
from ai.decorators.logging_decorator import configure_logging
from ai.decorators.timing_decorator import TimingDecorator
from ai.interfaces import IEstrategiaConPerfil, IRecommendationStrategy, UserFeatures
from ai.model_factory import ModelFactory
from ai.strategies import load_feature_table
from application.autenticacion import ServicioAutenticacion
from application.buscador import BuscadorPeliculas
from application.catalogo_es import GENEROS_ES, CatalogoEspanol
from application.correo import enviador_desde
from application.error_journal import ErrorJournal
from application.interaction_ingestion_service import InteractionIngestionService
from application.metricas import resumen_arranque_en_frio, resumen_de_evaluacion
from application.recommendation_controller import RecommendationController
from application.tendencias import ContadorTendencias
from config.settings import Settings, load_settings
from data.feature_store_manager import ParquetFeatureStoreManager
from errors import (
    MENSAJES_ES,
    AccesoDenegadoError,
    ModelUnavailableError,
    RecommendationPlatformError,
    SesionInvalidaError,
    detalle_de_validacion,
)
from messaging.feedback_history_service import FeedbackHistoryService
from messaging.interaction_event_queue import RetryBuffer, build_event_queue
from persistence.nosql_interaction_history_repository import NoSqlInteractionHistoryRepository
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_repositorio_cuentas import SqlRepositorioCuentas
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from presentation.schemas import (
    MAX_K,
    ActivacionDobleFactor,
    CodigoRequest,
    CodigosRespaldo,
    CompletarDobleFactorRequest,
    ConfirmarCorreoRequest,
    DobleFactorRequerido,
    EliminarCuentaRequest,
    ErrorResponse,
    Genero,
    HealthResponse,
    InicioSesionRequest,
    InteractionAccepted,
    InteractionRequest,
    PeliculasParecidas,
    RecommendationResponse,
    RecommendedItem,
    ReenviarCodigoRequest,
    RegistroRequest,
    SesionIniciada,
    TendenciasResponse,
    VerificacionPendiente,
)

#: Static assets of the Presentation Layer, served at /ui when present.
STATIC_DIR = FilePath(__file__).resolve().parents[1] / "presentation" / "static"

#: Películas por género que se muestran en el registro. Con 3 géneros x 16 la
#: evaluación offline dio la mejor precisión inicial (reports/arranque_en_frio.json).
PELICULAS_POR_GENERO = 16

ENCABEZADOS_DE_SEGURIDAD = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
# Solo scripts y estilos propios (más Chart.js); imágenes propias y los pósters de TMDB.
POLITICA_DE_CONTENIDO = (
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://image.tmdb.org; style-src 'self'; frame-ancestors 'none'"
)


def usuario_autenticado(request: Request) -> int:
    """Id del usuario dueño del token ``Authorization: Bearer ...``."""
    encabezado = request.headers.get("Authorization", "")
    if not encabezado.startswith("Bearer "):
        raise SesionInvalidaError("missing bearer token")
    servicio = request.app.state.components.autenticacion
    return servicio.usuario_del_token(encabezado.removeprefix("Bearer "))


def verificar_dueno(request: Request, user_id: int) -> None:
    """Un usuario solo puede pedir recomendaciones o registrar eventos a su nombre."""
    if usuario_autenticado(request) != user_id:
        raise AccesoDenegadoError(f"token does not belong to user {user_id}")


@dataclass
class ApplicationComponents:
    """Everything the composition root built, kept for the routes and /health."""

    settings: Settings
    controller: RecommendationController
    ingestion: InteractionIngestionService
    feature_store: ParquetFeatureStoreManager
    catalog: SqlCatalogRepository
    profiles: SqlUserProfileRepository
    history: NoSqlInteractionHistoryRepository
    worker: FeedbackHistoryService
    queue_backend: str
    cache_backend: str
    timing: TimingDecorator | None
    active_strategy: str
    autenticacion: ServicioAutenticacion
    tendencias: ContadorTendencias
    catalogo_es: CatalogoEspanol
    evaluacion: dict[str, Any] | None
    arranque_en_frio: dict[str, Any] | None
    startup_error: str | None = None
    journal: ErrorJournal | None = None


def build_components(settings: Settings | None = None) -> ApplicationComponents:
    """Wire the whole application from configuration, once."""
    settings = settings or load_settings()
    logger = configure_logging(settings)

    feature_store = ParquetFeatureStoreManager.from_settings(settings)
    catalog = SqlCatalogRepository.from_settings(settings)
    profiles = SqlUserProfileRepository.from_settings(settings)
    history = NoSqlInteractionHistoryRepository.from_settings(settings)
    queue = build_event_queue(settings)
    journal = ErrorJournal(settings.logging.error_journal_path)
    tendencias = ContadorTendencias(
        settings.web.minutos_tendencia, sexo_de=lambda user_id: _sexo_de(profiles, user_id)
    )
    catalogo_es = CatalogoEspanol.desde_csv(settings.web.catalogo_es)
    autenticacion = ServicioAutenticacion(
        SqlRepositorioCuentas.from_settings(settings),
        profiles,
        secreto=settings.web.jwt_secreto,
        minutos_sesion=settings.web.minutos_sesion,
        max_intentos=settings.web.max_intentos_fallidos,
        segundos_bloqueo=settings.web.segundos_bloqueo,
        enviador=enviador_desde(settings),
        minutos_codigo_correo=settings.web.minutos_codigo_correo,
    )

    fallback, fallback_error = _build_fallback(settings)
    buscador = _build_buscador(catalog, catalogo_es, fallback)
    base, active_strategy, startup_error = _build_base_strategy(settings, fallback)
    decorated = build_chain(base, settings.model.decorator_chain, settings=settings)
    # The fallback runs through the same decorator chain as the active
    # strategy, so a cold-start or degraded request is cached, timed and
    # logged exactly like a normal one (Section 9.9: one operational record
    # per prediction, whichever strategy produced it).
    if fallback is not None:
        fallback = build_chain(fallback, settings.model.decorator_chain, settings=settings)

    controller = RecommendationController(
        strategy=decorated,
        feature_store=feature_store,
        event_queue=queue,
        catalog_repository=catalog,
        profile_repository=profiles,
        fallback_strategy=fallback,
        settings=settings,
        logger=logger,
        error_journal=journal,
        # When the configured strategy could not be loaded, the fallback is
        # serving in its place for the whole life of the process; the client
        # is told so on every response, not only in /health.
        startup_degradation_reason=(ModelUnavailableError.code if startup_error else None),
        history_repository=history,
        estrategia_calentamiento=_build_calentamiento(settings),
        catalogo_es=catalogo_es,
        tendencias=tendencias,
        buscador=buscador,
        mas_vistas_por_genero=_mas_vistas_por_genero(settings),
    )

    return ApplicationComponents(
        settings=settings,
        controller=controller,
        ingestion=InteractionIngestionService(
            queue, logger=logger, observadores=[tendencias.registrar]
        ),
        feature_store=feature_store,
        catalog=catalog,
        profiles=profiles,
        history=history,
        worker=FeedbackHistoryService(
            queue,
            history,
            logger=logger,
            retry_buffer=RetryBuffer(settings.messaging.retry_buffer_path),
        ),
        queue_backend=queue.describe(),
        cache_backend=_find_cache_backend(decorated),
        timing=_find_timing(decorated),
        active_strategy=active_strategy,
        startup_error=startup_error or fallback_error,
        journal=journal,
        autenticacion=autenticacion,
        tendencias=tendencias,
        catalogo_es=catalogo_es,
        evaluacion=resumen_de_evaluacion(settings.web.reporte_evaluacion),
        arranque_en_frio=resumen_arranque_en_frio(settings.web.reporte_arranque_en_frio),
    )


def _build_buscador(
    catalog: SqlCatalogRepository,
    catalogo_es: CatalogoEspanol,
    popularidad: IRecommendationStrategy | None,
) -> BuscadorPeliculas:
    """Índice de títulos ordenado con el mismo ranking de popularidad del arranque en frío."""
    titulos = catalog.listar_titulos()
    orden: list[int] = []
    if popularidad is not None:
        anonimo = UserFeatures(user_id=0, is_cold_start=True)
        orden = [s.item_id for s in popularidad.predict(anonimo, k=len(titulos))]
    return BuscadorPeliculas(titulos, catalogo_es.titulos(), orden)


def _mas_vistas_por_genero(settings: Settings) -> dict[str, list[int]]:
    """Las películas con más interacciones de cada género, para el registro."""
    tabla = load_feature_table(
        settings.feature_store.version_dir, "category_popularity", required=False
    )
    if tabla is None:
        return {}
    tabla = tabla.sort_values("interaction_count", ascending=False)
    return {
        str(genero): grupo["item_id"].astype(int).tolist()[:PELICULAS_POR_GENERO]
        for genero, grupo in tabla.groupby("category")
    }


def _sexo_de(profiles: SqlUserProfileRepository, user_id: int) -> str | None:
    perfil = profiles.get_by_id(user_id)
    return perfil.attributes.get("sexo") if perfil else None


def _build_calentamiento(settings: Settings) -> IEstrategiaConPerfil | None:
    """ALS con fold-in para usuarios nuevos; sin él se quedan en popularidad."""
    try:
        estrategia = ModelFactory.create("als", settings)
    except ModelUnavailableError:
        return None
    return estrategia if isinstance(estrategia, IEstrategiaConPerfil) else None


class _UnavailableStrategy(IRecommendationStrategy):
    """Null object used when neither the model nor the fallback can be built.

    Without it, a machine with no materialised artifacts would fail at
    startup, and a service that cannot start is not the controlled
    degradation NFR-05 asks for. With it, the service starts, ``/health``
    reports ``degraded``, and a recommendation request is answered with an
    explicit translated error instead of a crash.
    """

    strategy_id = "unavailable"

    def __init__(self, reason: str) -> None:
        self._reason = reason
        self._model_version = "none"

    def predict(self, user_features: Any, k: int = 10) -> list[Any]:
        raise ModelUnavailableError(
            "no recommendation strategy could be loaded", detail=self._reason
        )


def _build_base_strategy(
    settings: Settings, fallback: IRecommendationStrategy | None
) -> tuple[IRecommendationStrategy, str, str | None]:
    """Resolve the active strategy, degrading to the fallback if it cannot load.

    A model that fails to load must not stop the service from starting:
    NFR-05 requires catalog navigation to survive a failure of the
    recommendation subsystem.
    """
    try:
        base = ModelFactory.create(settings.model.active_strategy, settings)
        return base, settings.model.active_strategy, None
    except ModelUnavailableError as exc:
        reason = f"{exc.code}: {exc.message}"
        if fallback is None:
            return _UnavailableStrategy(reason), "unavailable", reason
        return fallback, settings.model.fallback_strategy, reason


def _build_fallback(settings: Settings) -> tuple[IRecommendationStrategy | None, str | None]:
    """Build the cold-start fallback, tolerating its absence."""
    try:
        return ModelFactory.create(settings.model.fallback_strategy, settings), None
    except ModelUnavailableError as exc:
        return None, f"fallback unavailable: {exc.message}"


def _find_timing(strategy: Any) -> TimingDecorator | None:
    """Locate the timing decorator in the chain, to report the served p95."""
    while strategy is not None:
        if isinstance(strategy, TimingDecorator):
            return strategy
        strategy = getattr(strategy, "wrapped", None)
    return None


def _find_cache_backend(strategy: Any) -> str:
    while strategy is not None:
        if isinstance(strategy, CachingDecorator):
            return strategy.backend.describe()
        strategy = getattr(strategy, "wrapped", None)
    return "none"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application with its routes and exception handlers."""
    components = build_components(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # With a real broker the worker is its own process (Section 5.2). With
        # the in-process queue there is nothing else to consume it, so the
        # worker runs on a daemon thread and the asynchronous path of FR-06
        # stays exercised.
        if components.queue_backend == "in-memory":
            components.worker.start()
        try:
            yield
        finally:
            components.worker.stop()

    app = FastAPI(
        title="Plataforma de Recomendación Inteligente",
        version="1.0.0",
        summary="Entrega final: aplicación web sobre la arquitectura en capas del Milestone 3",
        lifespan=lifespan,
    )
    app.state.components = components

    @app.middleware("http")
    async def _encabezados_de_seguridad(request: Request, call_next: Any) -> Any:
        respuesta = await call_next(request)
        respuesta.headers.update(ENCABEZADOS_DE_SEGURIDAD)
        # Swagger (/docs) usa scripts en línea; la política estricta es para la interfaz.
        if not request.url.path.startswith(("/docs", "/redoc")):
            respuesta.headers["Content-Security-Policy"] = POLITICA_DE_CONTENIDO
        # Que el navegador pregunte siempre si la interfaz cambió; si no, recibe
        # un 304 sin contenido. Evita que se quede con un app.js viejo.
        if request.url.path.startswith("/ui"):
            respuesta.headers["Cache-Control"] = "no-cache"
        return respuesta

    def get_components() -> ApplicationComponents:
        return components

    def _journal(request: Request, code: str, message: str, status: int, **fields: Any) -> None:
        components.journal.record(  # type: ignore[union-attr]
            "error", code, message, route=request.url.path, status=status, **fields
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Table 10, row 1: a schema violation is a 400, not the default 422."""
        first = exc.errors()[0] if exc.errors() else {}
        detail = detalle_de_validacion(first) if first else None
        message = MENSAJES_ES["validation_error"]
        # the journal keeps Pydantic's original text for developers
        _journal(request, "validation_error", message, 400, detail=str(first.get("msg", "")))
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                code="validation_error", message=message, detail=detail
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def _on_value_error(request: Request, exc: ValueError) -> JSONResponse:
        message = MENSAJES_ES["validation_error"]
        # str(exc) is a developer message in English: journaled, not returned
        _journal(request, "validation_error", message, 400, detail=str(exc))
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                code="validation_error",
                message=message,
                detail="Revisa los parámetros de la solicitud.",
            ).model_dump(),
        )

    @app.exception_handler(RecommendationPlatformError)
    async def _on_platform_error(
        request: Request, exc: RecommendationPlatformError
    ) -> JSONResponse:
        """Translate the vocabulary of Table 10 into status codes, once."""
        status = exc.status_code if exc.status_code != 200 else 503
        payload = exc.to_payload()
        _journal(request, exc.code, payload["message"], status, detail=exc.detail)
        # the technical detail is English and stays in the journal; the client
        # only gets the Spanish message
        payload.pop("detail", None)
        return JSONResponse(status_code=status, content=payload)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Unknown routes and methods answer in Spanish, like every other error."""
        code = {404: "route_not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        message = MENSAJES_ES[code]
        _journal(request, code, message, exc.status_code, detail=str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=code, message=message).model_dump(exclude_none=True),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Last resort: an unexpected failure is a Spanish 500, never a traceback."""
        message = MENSAJES_ES["internal_error"]
        _journal(request, "internal_error", message, 500, detail=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(code="internal_error", message=message).model_dump(
                exclude_none=True
            ),
        )

    @app.get(
        "/recommendations/{user_id}",
        response_model=RecommendationResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
        summary="Recomendaciones Top-K para un usuario (FR-04, FR-05)",
        tags=["Recomendaciones"],
        description=(
            "Devuelve las k películas recomendadas para el usuario. Si el modelo o las "
            "características no están disponibles, responde con el ranking de popularidad "
            "y lo indica en degraded y degradation_reason."
        ),
    )
    def get_recommendations(
        request: Request,
        user_id: int = Path(ge=1, description="Identificador del usuario que solicita"),
        k: int = Query(default=10, ge=1, le=MAX_K, description="Tamaño del ranking"),
        parts: ApplicationComponents = Depends(get_components),
    ) -> RecommendationResponse:
        """The endpoint of Section 9.6: ``GET /recommendations/{user_id}?k=10``."""
        verificar_dueno(request, user_id)
        return parts.controller.recommend(user_id, k)

    @app.post(
        "/interactions",
        response_model=InteractionAccepted,
        status_code=202,
        responses={400: {"model": ErrorResponse}},
        summary="Registrar una interacción del usuario (FR-02)",
        tags=["Interacciones"],
        description=(
            "Encola el evento sin bloquear la navegación; responde 202 y el historial se "
            "guarda fuera de la solicitud."
        ),
    )
    def post_interaction(
        request: InteractionRequest,
        http: Request,
        parts: ApplicationComponents = Depends(get_components),
    ) -> InteractionAccepted:
        """Queue an interaction without blocking navigation."""
        verificar_dueno(http, request.user_id)
        return parts.ingestion.ingest(request)

    @app.post(
        "/auth/registro",
        response_model=VerificacionPendiente,
        status_code=201,
        summary="Crear una cuenta; se activa con el código que llega al correo",
        tags=["Cuentas"],
    )
    def registrar(
        datos: RegistroRequest, parts: ApplicationComponents = Depends(get_components)
    ) -> VerificacionPendiente:
        generos = [g for g in datos.generos if g in GENEROS_ES]
        return parts.autenticacion.registrar(
            datos.usuario,
            datos.correo,
            datos.contrasena,
            generos,
            sexo=datos.sexo,
            peliculas=datos.peliculas,
        )

    @app.post(
        "/auth/correo/confirmar",
        response_model=SesionIniciada,
        summary="Confirmar el correo con el código de 6 dígitos e iniciar sesión",
        tags=["Cuentas"],
    )
    def confirmar_correo(
        datos: ConfirmarCorreoRequest, parts: ApplicationComponents = Depends(get_components)
    ) -> SesionIniciada:
        return parts.autenticacion.confirmar_correo(datos.correo, datos.codigo)

    @app.post(
        "/auth/correo/reenviar",
        summary="Enviar otra vez el código de confirmación (máximo uno por minuto)",
        tags=["Cuentas"],
    )
    def reenviar_codigo(
        datos: ReenviarCodigoRequest, parts: ApplicationComponents = Depends(get_components)
    ) -> dict[str, bool]:
        parts.autenticacion.reenviar_codigo(datos.correo)
        return {"enviado": True}

    @app.post(
        "/auth/sesion",
        response_model=SesionIniciada | DobleFactorRequerido,
        summary="Iniciar sesión con correo y contraseña",
        description="Si la cuenta tiene verificación en dos pasos, responde con un token "
        "que se canjea en /auth/sesion/codigo junto con el código de la app.",
        tags=["Cuentas"],
    )
    def iniciar_sesion(
        datos: InicioSesionRequest, parts: ApplicationComponents = Depends(get_components)
    ) -> SesionIniciada | DobleFactorRequerido:
        return parts.autenticacion.iniciar_sesion(datos.correo, datos.contrasena)

    @app.post(
        "/auth/sesion/codigo",
        response_model=SesionIniciada,
        summary="Segundo paso: código de la app autenticadora o de respaldo",
        tags=["Cuentas"],
    )
    def completar_sesion(
        datos: CompletarDobleFactorRequest, parts: ApplicationComponents = Depends(get_components)
    ) -> SesionIniciada:
        return parts.autenticacion.completar_doble_factor(datos.token_doble_factor, datos.codigo)

    @app.post(
        "/auth/doble-factor",
        response_model=ActivacionDobleFactor,
        summary="Empezar a activar la verificación en dos pasos (devuelve el QR)",
        tags=["Cuentas"],
    )
    def alta_doble_factor(
        request: Request, parts: ApplicationComponents = Depends(get_components)
    ) -> ActivacionDobleFactor:
        return parts.autenticacion.iniciar_alta_doble_factor(usuario_autenticado(request))

    @app.post(
        "/auth/doble-factor/confirmar",
        response_model=CodigosRespaldo,
        summary="Confirmar con el primer código y activar; devuelve los códigos de respaldo",
        tags=["Cuentas"],
    )
    def confirmar_doble_factor(
        datos: CodigoRequest,
        request: Request,
        parts: ApplicationComponents = Depends(get_components),
    ) -> CodigosRespaldo:
        servicio = parts.autenticacion
        codigos = servicio.confirmar_doble_factor(usuario_autenticado(request), datos.codigo)
        return CodigosRespaldo(codigos=codigos)

    @app.post(
        "/auth/doble-factor/desactivar",
        summary="Desactivar la verificación en dos pasos (pide un código)",
        tags=["Cuentas"],
    )
    def desactivar_doble_factor(
        datos: CodigoRequest,
        request: Request,
        parts: ApplicationComponents = Depends(get_components),
    ) -> dict[str, bool]:
        parts.autenticacion.desactivar_doble_factor(usuario_autenticado(request), datos.codigo)
        return {"doble_factor_activo": False}

    @app.post(
        "/auth/cuenta/eliminar",
        summary="Eliminar mi cuenta (pide la contraseña y, si aplica, el código)",
        tags=["Cuentas"],
    )
    def eliminar_cuenta(
        datos: EliminarCuentaRequest,
        request: Request,
        parts: ApplicationComponents = Depends(get_components),
    ) -> dict[str, bool]:
        parts.autenticacion.eliminar_cuenta(
            usuario_autenticado(request), datos.contrasena, datos.codigo
        )
        return {"cuenta_eliminada": True}

    @app.get("/auth/yo", summary="Datos del usuario de la sesión actual", tags=["Cuentas"])
    def quien_soy(
        request: Request, parts: ApplicationComponents = Depends(get_components)
    ) -> dict[str, Any]:
        user_id = usuario_autenticado(request)
        perfil = parts.profiles.get_by_id(user_id)
        generos = perfil.preferred_categories if perfil else ()
        cuenta = parts.autenticacion.cuenta(user_id)
        return {
            "user_id": user_id,
            "usuario": cuenta.usuario if cuenta else None,
            "correo": cuenta.correo if cuenta else None,
            "doble_factor_activo": bool(cuenta and cuenta.totp_activo),
            "generos": [GENEROS_ES.get(g, g) for g in generos],
            "generos_clave": list(generos),
            "me_gusta": parts.controller.peliculas_gustadas(user_id),
        }

    @app.get(
        "/generos",
        response_model=list[Genero],
        summary="Géneros que se pueden elegir al registrarse",
        tags=["Catálogo"],
    )
    def generos() -> list[Genero]:
        return [Genero(clave=c, nombre=n) for c, n in GENEROS_ES.items() if c != "IMAX"]

    @app.get(
        "/tendencias",
        response_model=TendenciasResponse,
        summary="Top-K de lo más popular del momento",
        tags=["Recomendaciones"],
    )
    def tendencias(
        k: int = Query(default=10, ge=1, le=MAX_K),
        parts: ApplicationComponents = Depends(get_components),
    ) -> TendenciasResponse:
        items, en_vivo = parts.controller.tendencias(k)
        return TendenciasResponse(
            items=items, ventana_minutos=parts.tendencias.minutos, en_vivo=en_vivo
        )

    @app.get(
        "/peliculas/para-registro",
        response_model=list[RecommendedItem],
        summary="Películas populares de los géneros elegidos, para marcarlas al registrarse",
        tags=["Catálogo"],
    )
    def peliculas_para_registro(
        generos: list[str] = Query(default=[], max_length=3),
        parts: ApplicationComponents = Depends(get_components),
    ) -> list[RecommendedItem]:
        validos = [g for g in generos if g in GENEROS_ES]
        return parts.controller.peliculas_para_registro(validos)

    @app.get(
        "/peliculas/populares",
        response_model=list[RecommendedItem],
        summary="Las más vistas de un género",
        tags=["Catálogo"],
    )
    def populares_de_genero(
        genero: str = Query(max_length=40),
        k: int = Query(default=PELICULAS_POR_GENERO, ge=1, le=PELICULAS_POR_GENERO),
        parts: ApplicationComponents = Depends(get_components),
    ) -> list[RecommendedItem]:
        return parts.controller.populares_de_genero(genero, k)

    @app.get(
        "/peliculas/{item_id}/parecidas",
        response_model=PeliculasParecidas,
        summary="Películas que suelen gustarle a quien le gustó esta",
        tags=["Catálogo"],
    )
    def peliculas_parecidas(
        item_id: int = Path(ge=1),
        k: int = Query(default=12, ge=1, le=MAX_K),
        parts: ApplicationComponents = Depends(get_components),
    ) -> PeliculasParecidas:
        controlador = parts.controller
        return PeliculasParecidas(
            base=controlador.ficha(item_id), items=controlador.parecidas(item_id, k)
        )

    @app.get(
        "/peliculas/buscar",
        response_model=list[RecommendedItem],
        summary="Buscar películas por título (original o en español)",
        tags=["Catálogo"],
    )
    def buscar_peliculas(
        q: str = Query(min_length=1, max_length=100, description="Texto a buscar"),
        k: int = Query(default=20, ge=1, le=MAX_K),
        parts: ApplicationComponents = Depends(get_components),
    ) -> list[RecommendedItem]:
        return parts.controller.buscar(q, k)

    @app.get("/metricas", summary="Métricas offline y del servicio en vivo", tags=["Estado"])
    def metricas(parts: ApplicationComponents = Depends(get_components)) -> dict[str, Any]:
        p95 = parts.controller.latencia_p95()
        return {
            "evaluacion": parts.evaluacion,
            "servicio": {
                "estrategia_activa": parts.active_strategy,
                "latencia_p95_ms": None if p95 is None else round(p95, 2),
                "usuarios_activos": parts.autenticacion.usuarios_activos(),
                "interacciones_por_minuto": parts.tendencias.actividad_por_minuto(),
                "catalogo_con_tmdb": parts.catalogo_es.con_tmdb,
            },
            "arranque_en_frio": parts.arranque_en_frio,
            "por_sexo": {
                sexo: [{"titulo": i.titulo, "puntos": i.score} for i in items]
                for sexo, items in parts.controller.tendencias_por_sexo(5).items()
            },
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Estado operativo del servicio",
        tags=["Estado"],
        description="Estrategia activa, backends en uso y tamaño del catálogo.",
    )
    def health(parts: ApplicationComponents = Depends(get_components)) -> HealthResponse:
        return HealthResponse(
            status="degraded" if parts.startup_error else "ok",
            active_strategy=parts.active_strategy,
            registered_strategies=list(ModelFactory.available()),
            decorator_chain=list(parts.settings.model.decorator_chain),
            feature_store_version=parts.feature_store.version,
            feature_store_available=parts.feature_store.is_available(),
            cache_backend=parts.cache_backend,
            queue_backend=parts.queue_backend,
            history_backend=parts.history.backend,
            catalog_items=_safe_count(parts.catalog.count),
            user_profiles=_safe_count(parts.profiles.count),
        )

    _mount_user_interface(app)
    return app


def _mount_user_interface(app: FastAPI) -> None:
    """Serve the Presentation Layer page, when it is present.

    Section 5.2 describes the Presentation Layer as the part that sends
    recommendation requests and shows the results, and states that it holds no
    business or AI logic. The page mounted here is exactly that: static assets
    that call the same public endpoints any other client would, over the same
    origin, so no component of the inner layers is aware of it.

    Serving it is optional. If the directory is absent the API still starts,
    which keeps the deployment of the service independent from the interface.
    """
    if not STATIC_DIR.is_dir():
        return

    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def _index() -> RedirectResponse:
        return RedirectResponse(url="/ui/")


def _safe_count(counter: Any) -> int:
    try:
        return int(counter())
    except Exception:  # noqa: BLE001 - /health must answer even with a broken store
        return -1


app = create_app()


__all__ = ["ApplicationComponents", "app", "build_components", "create_app"]
