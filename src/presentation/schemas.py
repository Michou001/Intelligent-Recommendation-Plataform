"""Layer 1 - Presentation: request and response contracts (Table 7).

Declarative Pydantic v2 schemas for user identifiers, ranking sizes, and
interaction payloads. Because FastAPI validates against them before the
route body runs, Fail Fast (Section 3.6, NFR-04) is a framework feature here
rather than hand-written code: a malformed request is rejected before any
component of the AI layer is invoked.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

#: Ranking sizes accepted by the endpoint. The upper bound exists so a single
#: request cannot ask the service to rank the whole catalog.
MIN_K = 1
MAX_K = 100

UserIdField = Annotated[int, Field(ge=1, description="Identificador del usuario que solicita")]
KField = Annotated[int, Field(ge=MIN_K, le=MAX_K, description="Cantidad de elementos a devolver")]


class RecommendedItem(BaseModel):
    """One entry of the Top-K ranking returned to the client."""

    item_id: int = Field(description="Identificador del elemento en el catálogo")
    rank: int = Field(ge=1, description="Posición en el ranking; 1 es la mejor")
    score: float = Field(description="Puntuación de relevancia de la estrategia activa")
    title: str | None = Field(default=None, description="Título en el catálogo, si se conoce")
    categories: list[str] = Field(
        default_factory=list, description="Géneros o categorías del contenido"
    )
    titulo: str | None = Field(default=None, description="Título en español, o el original")
    anio: int | None = Field(default=None, description="Año de estreno")
    generos: list[str] = Field(default_factory=list, description="Géneros en español")
    poster: str | None = Field(default=None, description="URL del póster (TMDB), si existe")
    sinopsis: str | None = Field(default=None, description="Sinopsis en español, si existe")


class RecommendationResponse(BaseModel):
    """Response of ``GET /recommendations/{user_id}``.

    ``degraded`` and ``degradation_reason`` make the controlled degradation of
    NFR-05 visible to the client: the request still succeeds, and the client
    can tell that it was served by the fallback rather than by the configured
    model.
    """

    user_id: int
    k: int
    items: list[RecommendedItem]
    strategy_id: str = Field(description="Estrategia que generó el ranking")
    model_version: str
    feature_store_version: str
    degraded: bool = Field(
        default=False, description="Verdadero si se usó la estrategia de respaldo"
    )
    degradation_reason: str | None = Field(
        default=None, description="Código del error que activó el respaldo, si hubo degradación"
    )
    latency_ms: float = Field(
        default=0.0, description="Latencia de la solicitud medida en el servidor"
    )
    fase: str = Field(
        default="personalizado",
        description="arranque_en_frio, calentamiento o personalizado: qué tanto conoce al usuario",
    )


#: Validación sencilla de correo: algo@dominio.ext, sin espacios.
_CORREO = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalizar_correo(valor: str) -> str:
    valor = valor.strip().lower()
    if not _CORREO.match(valor):
        raise ValueError("no tiene el formato nombre@dominio.com")
    return valor


#: Política de contraseñas acordada por el equipo para el registro.
LARGO_MINIMO_CONTRASENA = 8


def _es_simbolo(caracter: str) -> bool:
    return not caracter.isalnum() and not caracter.isspace() and caracter != "-"


def _unir(partes: list[str]) -> str:
    return partes[0] if len(partes) == 1 else f"{', '.join(partes[:-1])} y {partes[-1]}"


def faltas_de_contrasena(valor: str) -> list[str]:
    """Lo que le falta a una contraseña nueva; la lista vacía significa que cumple.

    Debe tener al menos 8 caracteres, una mayúscula, un número y un símbolo, y
    no puede llevar espacios ni guiones. La interfaz web aplica la misma regla
    mientras se escribe, pero la que manda es esta, en el servidor.
    """
    faltas = []
    if len(valor) < LARGO_MINIMO_CONTRASENA:
        faltas.append(f"al menos {LARGO_MINIMO_CONTRASENA} caracteres")
    if not any(c.isupper() for c in valor):
        faltas.append("una mayúscula")
    if not any(c.isdigit() for c in valor):
        faltas.append("un número")
    if not any(_es_simbolo(c) for c in valor):
        faltas.append("un símbolo (como ! # $ * o _)")
    return faltas


def validar_contrasena(valor: str) -> str:
    """Devuelve la contraseña si cumple la política; si no, ``ValueError`` en español."""
    if any(c.isspace() or c == "-" for c in valor):
        raise ValueError("no puede llevar espacios ni guiones (-)")
    faltas = faltas_de_contrasena(valor)
    if faltas:
        raise ValueError(f"debe llevar {_unir(faltas)}")
    return valor


class RegistroRequest(BaseModel):
    usuario: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    correo: str = Field(max_length=254, description="Único por cuenta; se guarda en minúsculas")
    contrasena: str = Field(min_length=8, max_length=128)
    generos: list[str] = Field(default_factory=list, max_length=3)
    sexo: Literal["mujer", "hombre", "otro"] | None = Field(
        default=None, description="Opcional; solo para la popularidad por sexo"
    )
    peliculas: list[int] = Field(
        default_factory=list, max_length=48, description="Películas que vio y le gustaron"
    )

    @field_validator("correo")
    @classmethod
    def _correo_valido(cls, valor: str) -> str:
        return _normalizar_correo(valor)

    @field_validator("contrasena")
    @classmethod
    def _contrasena_segura(cls, valor: str) -> str:
        return validar_contrasena(valor)


class InicioSesionRequest(BaseModel):
    correo: str = Field(max_length=254, description="El correo con el que se registró")
    contrasena: str = Field(min_length=1, max_length=128)

    @field_validator("correo")
    @classmethod
    def _correo_valido(cls, valor: str) -> str:
        return _normalizar_correo(valor)


class PeliculasParecidas(BaseModel):
    """Para la fila "Porque te gustó ...": la película de partida y sus parecidas."""

    base: RecommendedItem | None
    items: list[RecommendedItem]


class VerificacionPendiente(BaseModel):
    """La cuenta se creó, pero se activa hasta confirmar el código enviado al correo."""

    verificacion_requerida: bool = True
    correo: str
    minutos: int = Field(description="Minutos que sirve el código")


class ConfirmarCorreoRequest(BaseModel):
    correo: str = Field(max_length=254)
    codigo: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")

    @field_validator("correo")
    @classmethod
    def _correo_valido(cls, valor: str) -> str:
        return _normalizar_correo(valor)


class ReenviarCodigoRequest(BaseModel):
    correo: str = Field(max_length=254)

    @field_validator("correo")
    @classmethod
    def _correo_valido(cls, valor: str) -> str:
        return _normalizar_correo(valor)


class EliminarCuentaRequest(BaseModel):
    contrasena: str = Field(min_length=1, max_length=128)
    codigo: str = Field(default="", max_length=9, description="Solo con verificación en dos pasos")


class SesionIniciada(BaseModel):
    token: str
    user_id: int
    usuario: str
    minutos_sesion: int


class DobleFactorRequerido(BaseModel):
    """La contraseña fue correcta pero la cuenta pide además el código de la app."""

    doble_factor_requerido: bool = True
    token_doble_factor: str = Field(description="Sirve solo para el segundo paso; dura 5 minutos")


class CodigoRequest(BaseModel):
    codigo: str = Field(min_length=6, max_length=9, description="6 dígitos o código de respaldo")


class CompletarDobleFactorRequest(CodigoRequest):
    token_doble_factor: str = Field(min_length=1, max_length=1024)


class ActivacionDobleFactor(BaseModel):
    secreto: str = Field(description="Por si la app no puede escanear el QR")
    qr_svg: str


class CodigosRespaldo(BaseModel):
    codigos: list[str] = Field(description="Se muestran una sola vez; cada uno sirve una vez")


class Genero(BaseModel):
    clave: str = Field(description="Nombre del género en el catálogo (MovieLens)")
    nombre: str = Field(description="Nombre en español para mostrar")


class TendenciasResponse(BaseModel):
    items: list[RecommendedItem]
    ventana_minutos: int
    en_vivo: bool = Field(
        description="Falso si no hubo actividad reciente y se muestra la popularidad histórica"
    )


class InteractionRequest(BaseModel):
    """Payload of ``POST /interactions`` (FR-02).

    Validated before the event reaches the queue, so malformed interaction
    data is rejected at the entry point instead of poisoning the history.
    """

    user_id: UserIdField
    event_type: str = Field(
        min_length=1, max_length=64, description="Tipo de evento: click, play, rating, ..."
    )
    item_id: int | None = Field(default=None, ge=1)
    recommended_items: list[int] = Field(default_factory=list, max_length=MAX_K)
    model_version: str | None = Field(default=None, max_length=64)
    metadata: dict[str, str] = Field(default_factory=dict)
    occurred_at: datetime | None = Field(default=None)

    @field_validator("event_type")
    @classmethod
    def _strip_event_type(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("no puede estar vacío")
        return cleaned

    @field_validator("recommended_items")
    @classmethod
    def _validate_items(cls, value: list[int]) -> list[int]:
        if any(item < 1 for item in value):
            raise ValueError("los identificadores deben ser positivos")
        return value


class InteractionAccepted(BaseModel):
    """Acknowledgement of an accepted interaction event."""

    event_id: str
    accepted: bool = True
    queued: bool = Field(
        description="Falso si el broker no aceptó el evento y quedó en el búfer de reintentos"
    )


class ErrorResponse(BaseModel):
    """Client-facing error payload, free of stack traces and internal ids."""

    code: str
    message: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Operational snapshot of the composed application."""

    status: str
    active_strategy: str
    registered_strategies: list[str]
    decorator_chain: list[str]
    feature_store_version: str
    feature_store_available: bool
    cache_backend: str
    queue_backend: str
    history_backend: str
    catalog_items: int
    user_profiles: int


__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "InteractionAccepted",
    "InteractionRequest",
    "MAX_K",
    "MIN_K",
    "RecommendationResponse",
    "RecommendedItem",
    "ConfirmarCorreoRequest",
    "PeliculasParecidas",
    "ReenviarCodigoRequest",
    "VerificacionPendiente",
]
