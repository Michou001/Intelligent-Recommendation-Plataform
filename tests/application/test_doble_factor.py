"""Correo único en el registro y verificación en dos pasos (TOTP)."""

from __future__ import annotations

from pathlib import Path

import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from application.api import create_app
from application.autenticacion import ServicioAutenticacion
from application.doble_factor import VerificadorTotp
from config.settings import Settings
from errors import (
    CodigoInvalidoError,
    CorreoYaRegistradoError,
    CredencialesInvalidasError,
    DemasiadosIntentosError,
    SesionInvalidaError,
)
from persistence import get_engine
from persistence.sql_repositorio_cuentas import SqlRepositorioCuentas
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from presentation.schemas import DobleFactorRequerido, SesionIniciada
from tests.conftest import EnviadorDePrueba, registrar_confirmado, registrar_por_http


class Reloj:
    def __init__(self) -> None:
        self.ahora = 1_800_000_000.0

    def __call__(self) -> float:
        return self.ahora

    def avanzar(self) -> None:
        self.ahora += 30  # siguiente código TOTP


@pytest.fixture
def reloj() -> Reloj:
    return Reloj()


@pytest.fixture
def servicio(settings: Settings, reloj: Reloj) -> ServicioAutenticacion:
    return ServicioAutenticacion(
        SqlRepositorioCuentas.from_settings(settings),
        SqlUserProfileRepository.from_settings(settings),
        secreto="secreto-de-prueba",
        enviador=EnviadorDePrueba(),
        max_intentos=3,
        verificador=VerificadorTotp(reloj=reloj),
    )


def con_doble_factor(servicio: ServicioAutenticacion, reloj: Reloj) -> tuple[int, str, list[str]]:
    """Registra a 'ana' y le activa la verificación; devuelve (id, secreto, respaldos)."""
    sesion = registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    alta = servicio.iniciar_alta_doble_factor(sesion.user_id)
    respaldos = servicio.confirmar_doble_factor(
        sesion.user_id, pyotp.TOTP(alta.secreto).at(reloj())
    )
    reloj.avanzar()
    return sesion.user_id, alta.secreto, respaldos


# ---------------------------------------------------------------- correo


def test_no_se_puede_repetir_el_correo(servicio: ServicioAutenticacion) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(CorreoYaRegistradoError):
        registrar_confirmado(servicio, "otra", "ANA@correo.com", "Clave#Segura1", [])


def test_se_puede_entrar_con_el_correo(servicio: ServicioAutenticacion) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    sesion = servicio.iniciar_sesion("Ana@Correo.com", "Clave#Segura1")
    assert isinstance(sesion, SesionIniciada)
    assert sesion.usuario == "ana"


def test_el_nombre_de_usuario_ya_no_sirve_para_entrar(
    servicio: ServicioAutenticacion, settings: Settings
) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(CredencialesInvalidasError):
        servicio.iniciar_sesion("ana", "Clave#Segura1")
    with TestClient(create_app(settings)) as cliente:
        respuesta = cliente.post(
            "/auth/sesion", json={"correo": "ana", "contrasena": "Clave#Segura1"}
        )
    assert respuesta.status_code == 400


def test_el_registro_rechaza_un_correo_mal_escrito(settings: Settings) -> None:
    with TestClient(create_app(settings)) as cliente:
        respuesta = cliente.post(
            "/auth/registro",
            json={"usuario": "ana", "correo": "ana-sin-arroba", "contrasena": "Clave#Segura1"},
        )
    assert respuesta.status_code == 400
    assert respuesta.json()["detail"] == "El correo: no tiene el formato nombre@dominio.com."


def test_una_base_vieja_recibe_las_columnas_nuevas(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'vieja.db').as_posix()}"
    vieja = create_engine(url)
    with vieja.begin() as conexion:
        conexion.execute(
            text(
                "CREATE TABLE cuentas (user_id INTEGER PRIMARY KEY, usuario VARCHAR(32) NOT NULL "
                "UNIQUE, hash_contrasena VARCHAR(256) NOT NULL)"
            )
        )
        conexion.execute(text("INSERT INTO cuentas VALUES (611, 'vieja', 'x$y')"))
    vieja.dispose()

    columnas = {c["name"] for c in inspect(get_engine(url)).get_columns("cuentas")}
    assert {"correo", "totp_secreto", "totp_activo", "codigos_respaldo"} <= columnas
    cuenta = SqlRepositorioCuentas(get_engine(url)).obtener(611)
    assert cuenta is not None and cuenta.correo is None and not cuenta.totp_activo


# ---------------------------------------------------------------- TOTP


def test_con_doble_factor_la_contrasena_sola_no_basta(
    servicio: ServicioAutenticacion, reloj: Reloj
) -> None:
    _, secreto, _ = con_doble_factor(servicio, reloj)
    paso1 = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1")
    assert isinstance(paso1, DobleFactorRequerido)
    # El token del primer paso no sirve como sesión.
    with pytest.raises(SesionInvalidaError):
        servicio.usuario_del_token(paso1.token_doble_factor)

    sesion = servicio.completar_doble_factor(
        paso1.token_doble_factor, pyotp.TOTP(secreto).at(reloj())
    )
    assert servicio.usuario_del_token(sesion.token) == sesion.user_id


def test_un_codigo_no_se_puede_usar_dos_veces(
    servicio: ServicioAutenticacion, reloj: Reloj
) -> None:
    _, secreto, _ = con_doble_factor(servicio, reloj)
    codigo = pyotp.TOTP(secreto).at(reloj())
    token = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1").token_doble_factor
    servicio.completar_doble_factor(token, codigo)
    token = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1").token_doble_factor
    with pytest.raises(CodigoInvalidoError):
        servicio.completar_doble_factor(token, codigo)


def test_cada_codigo_de_respaldo_sirve_una_vez(
    servicio: ServicioAutenticacion, reloj: Reloj
) -> None:
    _, _, respaldos = con_doble_factor(servicio, reloj)
    assert len(respaldos) == 8
    token = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1").token_doble_factor
    servicio.completar_doble_factor(token, respaldos[0])
    token = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1").token_doble_factor
    with pytest.raises(CodigoInvalidoError):
        servicio.completar_doble_factor(token, respaldos[0])


def test_codigos_equivocados_bloquean_la_cuenta(
    servicio: ServicioAutenticacion, reloj: Reloj
) -> None:
    con_doble_factor(servicio, reloj)
    token = servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1").token_doble_factor
    for _ in range(3):
        with pytest.raises(CodigoInvalidoError):
            servicio.completar_doble_factor(token, "000000")
    with pytest.raises(DemasiadosIntentosError):
        servicio.completar_doble_factor(token, "000000")


def test_activar_exige_un_codigo_correcto(servicio: ServicioAutenticacion) -> None:
    sesion = registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    servicio.iniciar_alta_doble_factor(sesion.user_id)
    with pytest.raises(CodigoInvalidoError):
        servicio.confirmar_doble_factor(sesion.user_id, "123456")
    # Sin confirmar, la cuenta sigue entrando solo con la contraseña.
    assert isinstance(servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1"), SesionIniciada)


def test_desactivar(servicio: ServicioAutenticacion, reloj: Reloj) -> None:
    user_id, secreto, _ = con_doble_factor(servicio, reloj)
    servicio.desactivar_doble_factor(user_id, pyotp.TOTP(secreto).at(reloj()))
    assert isinstance(servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1"), SesionIniciada)


@pytest.mark.autenticacion
def test_flujo_completo_por_http(settings: Settings) -> None:
    with TestClient(create_app(settings)) as cliente:
        sesion = registrar_por_http(
            cliente,
            settings,
            {"usuario": "ana", "correo": "ana@correo.com", "contrasena": "Clave#Segura1"},
        )
        encabezado = {"Authorization": f"Bearer {sesion['token']}"}
        alta = cliente.post("/auth/doble-factor", headers=encabezado).json()
        assert alta["qr_svg"].startswith("<svg")
        codigo = pyotp.TOTP(alta["secreto"]).now()
        respaldos = cliente.post(
            "/auth/doble-factor/confirmar", json={"codigo": codigo}, headers=encabezado
        ).json()["codigos"]
        assert cliente.get("/auth/yo", headers=encabezado).json()["doble_factor_activo"] is True

        paso1 = cliente.post(
            "/auth/sesion", json={"correo": "ana@correo.com", "contrasena": "Clave#Segura1"}
        ).json()
        assert paso1["doble_factor_requerido"] is True
        paso2 = cliente.post(
            "/auth/sesion/codigo",
            json={"token_doble_factor": paso1["token_doble_factor"], "codigo": respaldos[0]},
        )
        assert paso2.status_code == 200
        assert paso2.json()["usuario"] == "ana"


# ---------------------------------------------------------------- eliminar la cuenta


def test_eliminar_la_cuenta_borra_correo_y_datos_del_perfil(
    servicio: ServicioAutenticacion, settings: Settings
) -> None:
    sesion = registrar_confirmado(
        servicio, "ana", "ana@correo.com", "Clave#Segura1", ["Drama"], sexo="mujer", peliculas=[1]
    )
    servicio.eliminar_cuenta(sesion.user_id, "Clave#Segura1")
    assert servicio.cuenta(sesion.user_id) is None
    perfil = SqlUserProfileRepository.from_settings(settings).get_by_id(sesion.user_id)
    assert perfil is not None and perfil.attributes == {} and perfil.preferred_categories == ()
    with pytest.raises(CredencialesInvalidasError):
        servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1")
    # el correo queda libre para registrarse otra vez
    registrar_confirmado(servicio, "ana2", "ana@correo.com", "Clave#Segura1", [])


def test_eliminar_la_cuenta_pide_la_contrasena(servicio: ServicioAutenticacion) -> None:
    sesion = registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(CredencialesInvalidasError):
        servicio.eliminar_cuenta(sesion.user_id, "otra-clave")
    assert servicio.cuenta(sesion.user_id) is not None


def test_con_doble_factor_eliminar_pide_tambien_el_codigo(
    servicio: ServicioAutenticacion, reloj: Reloj
) -> None:
    user_id, secreto, _ = con_doble_factor(servicio, reloj)
    with pytest.raises(CodigoInvalidoError):
        servicio.eliminar_cuenta(user_id, "Clave#Segura1", "000000")
    assert servicio.cuenta(user_id) is not None
    servicio.eliminar_cuenta(user_id, "Clave#Segura1", pyotp.TOTP(secreto).at(reloj()))
    assert servicio.cuenta(user_id) is None


@pytest.mark.autenticacion
def test_eliminar_la_cuenta_por_http(settings: Settings) -> None:
    with TestClient(create_app(settings)) as cliente:
        sesion = registrar_por_http(
            cliente,
            settings,
            {"usuario": "ana", "correo": "ana@correo.com", "contrasena": "Clave#Segura1"},
        )
        encabezado = {"Authorization": f"Bearer {sesion['token']}"}
        mal = cliente.post("/auth/cuenta/eliminar", json={"contrasena": "otra"}, headers=encabezado)
        assert mal.status_code == 401 and mal.json()["code"] == "credenciales_invalidas"
        bien = cliente.post(
            "/auth/cuenta/eliminar", json={"contrasena": "Clave#Segura1"}, headers=encabezado
        )
        assert bien.status_code == 200
        entrar = cliente.post(
            "/auth/sesion", json={"correo": "ana@correo.com", "contrasena": "Clave#Segura1"}
        )
        assert entrar.status_code == 401
