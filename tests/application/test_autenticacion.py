"""Registro, inicio de sesión y medidas de seguridad de la entrega final."""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from application.api import create_app
from application.autenticacion import (
    ServicioAutenticacion,
    cifrar_contrasena,
    contrasena_correcta,
)
from config.settings import Settings
from errors import (
    CredencialesInvalidasError,
    DemasiadosIntentosError,
    SesionInvalidaError,
    UsuarioYaExisteError,
)
from persistence.sql_repositorio_cuentas import SqlRepositorioCuentas
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from presentation.schemas import faltas_de_contrasena
from tests.conftest import EnviadorDePrueba, registrar_confirmado, registrar_por_http


@pytest.fixture
def servicio(settings: Settings) -> ServicioAutenticacion:
    return ServicioAutenticacion(
        SqlRepositorioCuentas.from_settings(settings),
        SqlUserProfileRepository.from_settings(settings),
        secreto="secreto-de-prueba",
        enviador=EnviadorDePrueba(),
        max_intentos=3,
    )


def test_la_contrasena_nunca_se_guarda_en_claro() -> None:
    guardado = cifrar_contrasena("mi-clave-secreta")
    assert "mi-clave-secreta" not in guardado
    assert contrasena_correcta("mi-clave-secreta", guardado)
    assert not contrasena_correcta("otra-clave", guardado)


def test_la_misma_contrasena_produce_hashes_distintos() -> None:
    assert cifrar_contrasena("igual") != cifrar_contrasena("igual")


def test_registrar_guarda_los_generos_como_perfil(
    servicio: ServicioAutenticacion, settings: Settings
) -> None:
    sesion = registrar_confirmado(
        servicio, "ana", "ana@correo.com", "Clave#Segura1", ["Comedy", "Drama"]
    )
    perfil = SqlUserProfileRepository.from_settings(settings).get_by_id(sesion.user_id)
    assert perfil is not None
    assert perfil.preferred_categories == ("Comedy", "Drama")
    assert servicio.usuario_del_token(sesion.token) == sesion.user_id


def test_el_registro_guarda_sexo_y_peliculas_elegidas(
    servicio: ServicioAutenticacion, settings: Settings
) -> None:
    sesion = registrar_confirmado(
        servicio,
        "ana",
        "ana@correo.com",
        "Clave#Segura1",
        ["Drama"],
        sexo="mujer",
        peliculas=[1, 2],
    )
    perfil = SqlUserProfileRepository.from_settings(settings).get_by_id(sesion.user_id)
    assert perfil is not None
    assert perfil.attributes == {"peliculas_iniciales": "1,2", "sexo": "mujer"}


def test_no_se_puede_repetir_el_nombre_de_usuario(servicio: ServicioAutenticacion) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(UsuarioYaExisteError):
        registrar_confirmado(servicio, "ana", "ana@correo.com", "Otra#Clave2", [])


def test_credenciales_incorrectas(servicio: ServicioAutenticacion) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(CredencialesInvalidasError):
        servicio.iniciar_sesion("ana@correo.com", "clave-equivocada")
    with pytest.raises(CredencialesInvalidasError):
        servicio.iniciar_sesion("nadie@correo.com", "Clave#Segura1")


def test_demasiados_intentos_bloquean_aunque_luego_acierte(
    servicio: ServicioAutenticacion,
) -> None:
    registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    for _ in range(3):
        with pytest.raises(CredencialesInvalidasError):
            servicio.iniciar_sesion("ana@correo.com", "mal")
    with pytest.raises(DemasiadosIntentosError):
        servicio.iniciar_sesion("ana@correo.com", "Clave#Segura1")


def test_un_token_alterado_o_vencido_no_sirve(servicio: ServicioAutenticacion) -> None:
    sesion = registrar_confirmado(servicio, "ana", "ana@correo.com", "Clave#Segura1", [])
    with pytest.raises(SesionInvalidaError):
        servicio.usuario_del_token(sesion.token[:-2] + "xx")
    vencido = jwt.encode(
        {"sub": str(sesion.user_id), "typ": "sesion", "exp": time.time() - 1},
        "secreto-de-prueba",
        "HS256",
    )
    with pytest.raises(SesionInvalidaError):
        servicio.usuario_del_token(vencido)


@pytest.mark.autenticacion
def test_la_api_exige_token_y_que_sea_del_mismo_usuario(settings: Settings) -> None:
    with TestClient(create_app(settings)) as cliente:
        assert cliente.get("/recommendations/1").status_code == 401

        sesion = registrar_por_http(
            cliente,
            settings,
            {"usuario": "ana", "correo": "ana@correo.com", "contrasena": "Clave#Segura1"},
        )
        encabezado = {"Authorization": f"Bearer {sesion['token']}"}
        otro_usuario = sesion["user_id"] + 1
        respuesta = cliente.get(f"/recommendations/{otro_usuario}", headers=encabezado)
        assert respuesta.status_code == 403

        propia = cliente.get(f"/recommendations/{sesion['user_id']}", headers=encabezado)
        assert propia.status_code == 200
        assert propia.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in propia.headers["Content-Security-Policy"]


@pytest.mark.autenticacion
def test_el_registro_rechaza_contrasenas_cortas_en_espanol(settings: Settings) -> None:
    with TestClient(create_app(settings)) as cliente:
        respuesta = cliente.post(
            "/auth/registro",
            json={"usuario": "ana", "correo": "ana@correo.com", "contrasena": "123"},
        )
    assert respuesta.status_code == 400
    assert respuesta.json()["detail"] == "La contraseña debe tener al menos 8 caracteres."


@pytest.mark.autenticacion
@pytest.mark.parametrize(
    ("contrasena", "detalle"),
    [
        ("12345678", "La contraseña: debe llevar una mayúscula y un símbolo (como ! # $ * o _)."),
        ("clave#segura1", "La contraseña: debe llevar una mayúscula."),
        ("Clave#Segura", "La contraseña: debe llevar un número."),
        ("ClaveSegura1", "La contraseña: debe llevar un símbolo (como ! # $ * o _)."),
        ("Clave Segura#1", "La contraseña: no puede llevar espacios ni guiones (-)."),
        ("Clave-Segura#1", "La contraseña: no puede llevar espacios ni guiones (-)."),
    ],
)
def test_el_registro_exige_una_contrasena_robusta(
    settings: Settings, contrasena: str, detalle: str
) -> None:
    with TestClient(create_app(settings)) as cliente:
        respuesta = cliente.post(
            "/auth/registro",
            json={"usuario": "ana", "correo": "ana@correo.com", "contrasena": contrasena},
        )
    assert respuesta.status_code == 400
    assert respuesta.json()["detail"] == detalle


def test_faltas_de_contrasena_dice_que_falta() -> None:
    assert faltas_de_contrasena("Clave#Segura1") == []
    assert faltas_de_contrasena("Ñandú_2026") == []  # letras con acento y guion bajo valen
    assert faltas_de_contrasena("abc") == [
        "al menos 8 caracteres",
        "una mayúscula",
        "un número",
        "un símbolo (como ! # $ * o _)",
    ]
