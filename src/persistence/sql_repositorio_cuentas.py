"""Repositorio SQL de las cuentas de acceso (entrega final)."""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError

from config.settings import Settings, load_settings
from errors import CorreoYaRegistradoError, UsuarioYaExisteError
from persistence import cuentas, get_engine, user_profiles
from persistence.interfaces import Cuenta, IRepositorioCuentas, UserId


class SqlRepositorioCuentas(IRepositorioCuentas):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # Dos registros simultáneos (varias máquinas) no deben recibir el mismo id.
        self._candado = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SqlRepositorioCuentas:
        settings = settings or load_settings()
        return cls(get_engine(settings.persistence.sql_url))

    def crear(
        self,
        usuario: str,
        hash_contrasena: str,
        correo: str | None = None,
        user_id: UserId | None = None,
        correo_verificado: bool = False,
    ) -> Cuenta:
        with self._candado, self._engine.begin() as conexion:
            mismo_usuario = select(cuentas.c.user_id).where(cuentas.c.usuario == usuario)
            if conexion.execute(mismo_usuario).first():
                raise UsuarioYaExisteError(f"el usuario {usuario!r} ya existe")
            if (
                correo
                and conexion.execute(
                    select(cuentas.c.user_id).where(cuentas.c.correo == correo)
                ).first()
            ):
                raise CorreoYaRegistradoError("the e-mail is already registered")
            if user_id is None:
                # Los ids nuevos empiezan después del último usuario de MovieLens.
                user_id = 1 + max(
                    conexion.execute(select(func.max(user_profiles.c.user_id))).scalar() or 0,
                    conexion.execute(select(func.max(cuentas.c.user_id))).scalar() or 0,
                )
            cuenta = Cuenta(
                user_id=user_id,
                usuario=usuario,
                hash_contrasena=hash_contrasena,
                correo=correo,
                correo_verificado=correo_verificado,
            )
            try:
                conexion.execute(
                    cuentas.insert().values(
                        user_id=cuenta.user_id,
                        usuario=cuenta.usuario,
                        hash_contrasena=cuenta.hash_contrasena,
                        correo=cuenta.correo,
                        correo_verificado=int(correo_verificado),
                    )
                )
            except IntegrityError as exc:  # respaldo: la base también exige que sean únicos
                raise UsuarioYaExisteError(f"account {usuario!r} already exists") from exc
        return cuenta

    def buscar_por_correo(self, correo: str) -> Cuenta | None:
        return self._una(select(cuentas).where(cuentas.c.correo == correo.strip().lower()))

    def obtener(self, user_id: UserId) -> Cuenta | None:
        return self._una(select(cuentas).where(cuentas.c.user_id == int(user_id)))

    def guardar_doble_factor(
        self, user_id: UserId, secreto: str | None, activo: bool, codigos_respaldo: str
    ) -> None:
        with self._engine.begin() as conexion:
            conexion.execute(
                cuentas.update()
                .where(cuentas.c.user_id == int(user_id))
                .values(
                    totp_secreto=secreto,
                    totp_activo=int(activo),
                    codigos_respaldo=codigos_respaldo,
                )
            )

    def guardar_codigo_correo(self, user_id: UserId, huella: str, expira: int) -> None:
        self._actualizar(user_id, codigo_correo=huella, codigo_correo_expira=expira)

    def marcar_correo_verificado(self, user_id: UserId) -> None:
        self._actualizar(
            user_id, correo_verificado=1, codigo_correo=None, codigo_correo_expira=None
        )

    def _actualizar(self, user_id: UserId, **valores: Any) -> None:
        with self._engine.begin() as conexion:
            conexion.execute(
                cuentas.update().where(cuentas.c.user_id == int(user_id)).values(**valores)
            )

    def eliminar(self, user_id: UserId) -> None:
        with self._engine.begin() as conexion:
            conexion.execute(cuentas.delete().where(cuentas.c.user_id == int(user_id)))

    def _una(self, consulta: Any) -> Cuenta | None:
        with self._engine.connect() as conexion:
            fila = conexion.execute(consulta).mappings().first()
        if fila is None:
            return None
        datos = dict(fila)
        datos["totp_activo"] = bool(datos["totp_activo"])
        datos["correo_verificado"] = bool(datos["correo_verificado"])
        return Cuenta(**datos)


__all__ = ["SqlRepositorioCuentas"]
