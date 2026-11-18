"""Exception hierarchy of Table 10 (Section 9.8).

The document specifies the exception types, the HTTP response, and the
recovery behaviour, but it does not assign them a module in the tree of
Section 9.2. They are declared here, at the source root, because they are
raised in four different layers (persistence, data, ai, messaging) and
translated in a single FastAPI handler in the application layer: placing them
inside any one layer would force the other layers to import from it and would
break the dependency direction rule.

``ValidationError`` is the sixth row of Table 10 and is *not* redefined here:
it is Pydantic's own exception, raised by the request schemas of the
presentation layer and translated by the same handler.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

#: Client-facing messages in Spanish, keyed by the stable error ``code``.
#: ``message`` on the exception stays in English for developers and logs;
#: only the payload returned to the client is translated. A code missing from
#: this table falls back to the original message, so nothing breaks when a
#: new exception is added before its translation.
MENSAJES_ES: dict[str, str] = {
    "validation_error": "La solicitud no cumple con el esquema declarado.",
    "user_not_found": "El usuario solicitado no existe.",
    "cold_start": "El usuario no tiene historial suficiente; se usó el ranking de popularidad.",
    "feature_store_unavailable": (
        "No se pudieron leer las características del usuario; se usó el ranking de popularidad."
    ),
    "model_unavailable": (
        "El modelo de recomendación no está disponible; se usó el ranking de popularidad."
    ),
    "event_publish_failed": (
        "El evento no pudo publicarse; quedó guardado en el búfer de reintentos."
    ),
    "internal_error": "Ocurrió un error interno del servicio.",
    "route_not_found": "La ruta solicitada no existe.",
    "method_not_allowed": "El método HTTP no está permitido en esta ruta.",
    "http_error": "La solicitud no pudo atenderse.",
    "usuario_ya_existe": "Ese nombre de usuario ya está registrado.",
    "correo_ya_registrado": "Ese correo ya tiene una cuenta.",
    "codigo_invalido": "El código de verificación no es válido o ya expiró.",
    "credenciales_invalidas": "Correo o contraseña incorrectos.",
    "correo_sin_verificar": "Confirma tu correo: te enviamos un código de 6 dígitos.",
    "correo_no_enviado": "No pudimos enviar el correo con tu código. Intenta de nuevo.",
    "sesion_invalida": "Tu sesión no es válida o ya expiró. Vuelve a iniciar sesión.",
    "acceso_denegado": "No tienes permiso para consultar los datos de otro usuario.",
    "demasiados_intentos": "Demasiados intentos fallidos. Espera un minuto e inténtalo de nuevo.",
}

#: Spanish names of the request fields, used to explain a validation error.
CAMPOS_ES: dict[str, str] = {
    "user_id": "El identificador de usuario",
    "k": "La cantidad de recomendaciones (k)",
    "event_type": "El tipo de evento",
    "item_id": "El identificador del elemento",
    "recommended_items": "La lista de elementos recomendados",
    "model_version": "La versión del modelo",
    "metadata": "Los metadatos",
    "occurred_at": "La fecha del evento",
    "usuario": "El nombre de usuario",
    "contrasena": "La contraseña",
    "correo": "El correo",
    "codigo": "El código de verificación",
    "generos": "La lista de géneros",
}

#: Spanish wording of the Pydantic error types a client can trigger. ``{...}``
#: placeholders are filled from the error context (bounds, lengths).
VALIDACIONES_ES: dict[str, str] = {
    "missing": "es obligatorio",
    "greater_than_equal": "debe ser mayor o igual que {ge}",
    "greater_than": "debe ser mayor que {gt}",
    "less_than_equal": "debe ser menor o igual que {le}",
    "less_than": "debe ser menor que {lt}",
    "int_parsing": "debe ser un número entero",
    "int_type": "debe ser un número entero",
    "int_from_float": "debe ser un número entero",
    "float_parsing": "debe ser un número",
    "string_type": "debe ser texto",
    "string_too_short": "debe tener al menos {min_length} caracteres",
    "string_too_long": "debe tener como máximo {max_length} caracteres",
    "string_pattern_mismatch": "solo admite letras, números y guion bajo",
    "too_long": "admite como máximo {max_length} elementos",
    "list_type": "debe ser una lista",
    "dict_type": "debe ser un objeto",
    "model_attributes_type": "debe ser un objeto JSON",
    "datetime_parsing": "debe ser una fecha y hora válida",
    "datetime_from_date_parsing": "debe ser una fecha y hora válida",
    "json_invalid": "no es un JSON válido",
    "value_error": "{error}",
}


def detalle_de_validacion(error: dict[str, Any]) -> str:
    """Explain one Pydantic/FastAPI validation error in Spanish for the client.

    The English text Pydantic produces is never returned: the error ``type``
    selects a Spanish template and ``loc`` names the offending field.
    """
    loc = [str(part) for part in error.get("loc", ()) if part not in ("body", "query", "path")]
    nombre = loc[-1] if loc else ""
    if nombre.isdigit() and len(loc) > 1:  # an element of a list, e.g. recommended_items.0
        nombre = loc[-2]
    campo = CAMPOS_ES.get(nombre, f"El campo «{nombre}»" if nombre else "La solicitud")
    plantilla = VALIDACIONES_ES.get(str(error.get("type", "")), "tiene un valor no válido")
    contexto = {key: str(value) for key, value in (error.get("ctx") or {}).items()}
    try:
        texto = plantilla.format(**contexto)
    except (KeyError, IndexError):
        texto = "tiene un valor no válido"
    return f"{campo}: {texto}." if plantilla == "{error}" else f"{campo} {texto}."


class RecommendationPlatformError(Exception):
    """Base class for every error the platform raises on its own behalf.

    Internal components never propagate library exceptions across a layer
    boundary; they translate them into one of the subclasses below, so the
    application layer depends on this vocabulary and not on Redis, pika,
    SQLAlchemy, or PyMongo error types (Section 9.8).
    """

    #: HTTP status the exception handler returns for this type.
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    #: Stable machine-readable code returned in the error payload.
    code: str = "internal_error"
    #: Whether the request can still be served with a degraded response.
    degradable: bool = False

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_payload(self) -> dict[str, str]:
        """Client-facing representation, free of stack traces and internal ids."""
        payload = {"code": self.code, "message": MENSAJES_ES.get(self.code, self.message)}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class UserNotFoundError(RecommendationPlatformError):
    """The user identifier does not exist in the profile repository (HTTP 404)."""

    status_code = HTTPStatus.NOT_FOUND
    code = "user_not_found"
    degradable = False


class ColdStartError(RecommendationPlatformError):
    """The user or item has no usable interaction history.

    HTTP 200: PopularityStrategy is used as a fallback (FR-05).
    """

    status_code = HTTPStatus.OK
    code = "cold_start"
    degradable = True


class FeatureStoreUnavailableError(RecommendationPlatformError):
    """The feature store cannot be read.

    HTTP 200 degraded: fallback to popularity ranking, logged at error level.
    """

    status_code = HTTPStatus.OK
    code = "feature_store_unavailable"
    degradable = True


class ModelUnavailableError(RecommendationPlatformError):
    """The configured model cannot be loaded or the inference fails.

    HTTP 200 degraded: fallback to popularity ranking, preserving catalog
    navigation (NFR-05).
    """

    status_code = HTTPStatus.OK
    code = "model_unavailable"
    degradable = True


class EventPublishError(RecommendationPlatformError):
    """The interaction event cannot be published to the queue.

    Does not affect the response: the event is written to a local retry
    buffer and the user response is unaffected.
    """

    status_code = HTTPStatus.OK
    code = "event_publish_failed"
    degradable = True


class UsuarioYaExisteError(RecommendationPlatformError):
    status_code = HTTPStatus.CONFLICT
    code = "usuario_ya_existe"


class CorreoYaRegistradoError(RecommendationPlatformError):
    status_code = HTTPStatus.CONFLICT
    code = "correo_ya_registrado"


class CodigoInvalidoError(RecommendationPlatformError):
    status_code = HTTPStatus.UNAUTHORIZED
    code = "codigo_invalido"


class CorreoNoEnviadoError(RecommendationPlatformError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "correo_no_enviado"


class CorreoSinVerificarError(RecommendationPlatformError):
    status_code = HTTPStatus.FORBIDDEN
    code = "correo_sin_verificar"


class CredencialesInvalidasError(RecommendationPlatformError):
    status_code = HTTPStatus.UNAUTHORIZED
    code = "credenciales_invalidas"


class SesionInvalidaError(RecommendationPlatformError):
    status_code = HTTPStatus.UNAUTHORIZED
    code = "sesion_invalida"


class AccesoDenegadoError(RecommendationPlatformError):
    status_code = HTTPStatus.FORBIDDEN
    code = "acceso_denegado"


class DemasiadosIntentosError(RecommendationPlatformError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    code = "demasiados_intentos"


#: Reason codes attached to a degraded response, so a client (and the tests)
#: can tell *why* the popularity fallback was served.
DEGRADATION_REASONS: dict[type[RecommendationPlatformError], str] = {
    ColdStartError: ColdStartError.code,
    FeatureStoreUnavailableError: FeatureStoreUnavailableError.code,
    ModelUnavailableError: ModelUnavailableError.code,
}

__all__ = [
    "MENSAJES_ES",
    "CAMPOS_ES",
    "VALIDACIONES_ES",
    "detalle_de_validacion",
    "DEGRADATION_REASONS",
    "ColdStartError",
    "EventPublishError",
    "FeatureStoreUnavailableError",
    "ModelUnavailableError",
    "RecommendationPlatformError",
    "UserNotFoundError",
    "AccesoDenegadoError",
    "CodigoInvalidoError",
    "CorreoNoEnviadoError",
    "CorreoSinVerificarError",
    "CorreoYaRegistradoError",
    "CredencialesInvalidasError",
    "DemasiadosIntentosError",
    "SesionInvalidaError",
    "UsuarioYaExisteError",
]
