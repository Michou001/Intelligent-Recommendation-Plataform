"""El correo se confirma con un código antes de que la cuenta dé acceso."""

from __future__ import annotations

import pytest

from application import autenticacion as modulo
from application.autenticacion import ServicioAutenticacion
from application.correo import BuzonLocal, EnviadorSmtp, enviador_desde
from config.settings import Settings
from errors import (
    CodigoInvalidoError,
    CorreoNoEnviadoError,
    CorreoSinVerificarError,
    CredencialesInvalidasError,
)
from persistence.sql_repositorio_cuentas import SqlRepositorioCuentas
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from presentation.schemas import SesionIniciada
from tests.conftest import EnviadorDePrueba


@pytest.fixture
def enviador() -> EnviadorDePrueba:
    return EnviadorDePrueba()


@pytest.fixture
def servicio(settings: Settings, enviador: EnviadorDePrueba) -> ServicioAutenticacion:
    return ServicioAutenticacion(
        SqlRepositorioCuentas.from_settings(settings),
        SqlUserProfileRepository.from_settings(settings),
        secreto="secreto-de-prueba",
        enviador=enviador,
        max_intentos=3,
    )


def test_registrarse_no_da_acceso_hasta_confirmar_el_correo(
    servicio: ServicioAutenticacion, enviador: EnviadorDePrueba
) -> None:
    pendiente = servicio.registrar("ana", "Ana@Correo.com", "Clave#Segura1", ["Drama"])
    assert pendiente.correo == "ana@correo.com" and pendiente.minutos == 10
    assert len(enviador.enviados) == 1
    with pytest.raises(CorreoSinVerificarError):
        servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1")

    sesion = servicio.confirmar_correo("ana@correo.com", enviador.codigo_para("ana@correo.com"))
    assert isinstance(sesion, SesionIniciada)
    assert isinstance(servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1"), SesionIniciada)


def test_un_codigo_equivocado_o_ya_usado_no_sirve(
    servicio: ServicioAutenticacion, enviador: EnviadorDePrueba
) -> None:
    servicio.registrar("ana", "ana@correo.com", "Clave#Segura1", [])
    codigo = enviador.codigo_para("ana@correo.com")
    with pytest.raises(CodigoInvalidoError):
        servicio.confirmar_correo("ana@correo.com", "000000" if codigo != "000000" else "111111")
    servicio.confirmar_correo("ana@correo.com", codigo)
    with pytest.raises(CodigoInvalidoError):
        servicio.confirmar_correo("ana@correo.com", codigo)


def test_el_codigo_vence(
    servicio: ServicioAutenticacion, enviador: EnviadorDePrueba, monkeypatch: pytest.MonkeyPatch
) -> None:
    servicio.registrar("ana", "ana@correo.com", "Clave#Segura1", [])
    codigo = enviador.codigo_para("ana@correo.com")
    ahora = modulo.time.time()
    monkeypatch.setattr(modulo.time, "time", lambda: ahora + 11 * 60)
    with pytest.raises(CodigoInvalidoError):
        servicio.confirmar_correo("ana@correo.com", codigo)


def test_una_cuenta_sin_confirmar_no_le_quita_el_correo_a_su_dueno(
    servicio: ServicioAutenticacion, enviador: EnviadorDePrueba
) -> None:
    servicio.registrar("intruso", "ana@correo.com", "clave-del-intruso", [])
    # La dueña real se registra con su correo: la cuenta sin confirmar se descarta.
    servicio.registrar("ana", "ana@correo.com", "clave-de-ana", [])
    sesion = servicio.confirmar_correo("ana@correo.com", enviador.codigo_para("ana@correo.com"))
    assert sesion.usuario == "ana"
    with pytest.raises(CredencialesInvalidasError):
        servicio.iniciar_sesion("ana@correo.com", "clave-del-intruso")


def test_el_reenvio_tiene_limite_y_no_revela_si_el_correo_existe(
    servicio: ServicioAutenticacion, enviador: EnviadorDePrueba
) -> None:
    servicio.registrar("ana", "ana@correo.com", "Clave#Segura1", [])
    servicio.reenviar_codigo("ana@correo.com")  # menos de un minuto: no manda otro
    servicio.reenviar_codigo("nadie@correo.com")  # no existe: tampoco falla
    assert len(enviador.enviados) == 1


def test_si_el_correo_no_sale_se_avisa_y_se_puede_reintentar(settings: Settings) -> None:
    class EnviadorCaido:
        def __init__(self) -> None:
            self.funciona = False
            self.enviados = 0

        def enviar(self, destinatario: str, asunto: str, texto: str) -> None:
            if not self.funciona:
                raise OSError("sin conexión")
            self.enviados += 1

    caido = EnviadorCaido()
    servicio = ServicioAutenticacion(
        SqlRepositorioCuentas.from_settings(settings),
        SqlUserProfileRepository.from_settings(settings),
        secreto="secreto-de-prueba",
        enviador=caido,  # type: ignore[arg-type]
    )
    with pytest.raises(CorreoNoEnviadoError):
        servicio.registrar("ana", "ana@correo.com", "Clave#Segura1", [])
    caido.funciona = True
    servicio.reenviar_codigo("ana@correo.com")  # no espera el minuto: el primero no salió
    assert caido.enviados == 1


def test_sin_smtp_el_codigo_va_al_buzon_local(settings: Settings) -> None:
    enviador = enviador_desde(settings)
    assert isinstance(enviador, BuzonLocal)
    enviador.enviar("ana@correo.com", "Tu código: 123456", "texto")
    assert "Para: ana@correo.com | Tu código: 123456" in settings.web.buzon_correos.read_text(
        encoding="utf-8"
    )


def test_el_envio_real_usa_tls_y_contrasena(monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas: list[str] = []

    class SmtpFalso:
        def __init__(self, host: str, puerto: int, timeout: int) -> None:
            llamadas.append(f"{host}:{puerto}")

        def __enter__(self) -> SmtpFalso:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def starttls(self, context: object) -> None:
            llamadas.append("starttls")

        def login(self, usuario: str, contrasena: str) -> None:
            llamadas.append(f"login {usuario}")

        def send_message(self, mensaje: object) -> None:
            llamadas.append(f"para {mensaje['To']}")  # type: ignore[index]

    monkeypatch.setattr("application.correo.smtplib.SMTP", SmtpFalso)
    EnviadorSmtp("smtp.gmail.com", 587, "equipo@gmail.com", "app-password", "").enviar(
        "ana@correo.com", "Asunto", "Texto"
    )
    assert llamadas == [
        "smtp.gmail.com:587",
        "starttls",
        "login equipo@gmail.com",
        "para ana@correo.com",
    ]
