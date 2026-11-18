"""Registro, inicio de sesión y tokens JWT de la aplicación web (entrega final).

Medidas de seguridad que se aplican aquí:
- La contraseña se guarda como hash scrypt con sal aleatoria, nunca en claro.
- La comparación del hash es de tiempo constante (hmac.compare_digest).
- Tras varios intentos fallidos el usuario queda bloqueado unos segundos.
- La sesión es un JWT firmado (HS256) que expira; el servidor no guarda sesiones.
- Verificación en dos pasos opcional con TOTP (ver doble_factor.py).
- El correo se confirma con un código de 6 dígitos que llega a ese correo; la
  cuenta no da acceso hasta confirmarlo.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from datetime import UTC, datetime, timedelta

import jwt

from application.correo import IEnviadorCorreo
from application.doble_factor import (
    VerificadorTotp,
    huella,
    nuevo_secreto,
    nuevos_codigos_respaldo,
    qr_para_alta,
)
from errors import (
    CodigoInvalidoError,
    CorreoNoEnviadoError,
    CorreoSinVerificarError,
    CredencialesInvalidasError,
    DemasiadosIntentosError,
    SesionInvalidaError,
)
from persistence.interfaces import Cuenta, IRepositorioCuentas, IUserProfileRepository, UserProfile
from presentation.schemas import (
    ActivacionDobleFactor,
    DobleFactorRequerido,
    SesionIniciada,
    VerificacionPendiente,
)

# Parámetros de scrypt equivalentes a la recomendación de OWASP (N=2^14, r=8, p=5).
_SCRYPT = {"n": 2**14, "r": 8, "p": 5}

#: Tiempo para escribir el código de la app después de dar la contraseña.
MINUTOS_PARA_EL_CODIGO = 5

#: Espera mínima entre dos correos de verificación a la misma cuenta.
SEGUNDOS_ENTRE_CORREOS = 60


def cifrar_contrasena(contrasena: str) -> str:
    sal = secrets.token_bytes(16)
    digesto = hashlib.scrypt(contrasena.encode(), salt=sal, **_SCRYPT)
    return f"{sal.hex()}${digesto.hex()}"


def contrasena_correcta(contrasena: str, guardado: str) -> bool:
    sal, digesto = guardado.split("$")
    calculado = hashlib.scrypt(contrasena.encode(), salt=bytes.fromhex(sal), **_SCRYPT)
    return hmac.compare_digest(calculado.hex(), digesto)


# Hash de relleno: si el usuario no existe se verifica contra él, para que el
# tiempo de respuesta no revele qué nombres de usuario están registrados.
_HASH_DE_RELLENO = cifrar_contrasena("relleno")


class ServicioAutenticacion:
    def __init__(
        self,
        cuentas: IRepositorioCuentas,
        perfiles: IUserProfileRepository,
        *,
        secreto: str,
        minutos_sesion: int = 60,
        max_intentos: int = 5,
        segundos_bloqueo: int = 60,
        enviador: IEnviadorCorreo,
        minutos_codigo_correo: int = 10,
        verificador: VerificadorTotp | None = None,
    ) -> None:
        self._cuentas = cuentas
        self._perfiles = perfiles
        self._secreto = secreto
        self._minutos_sesion = minutos_sesion
        self._max_intentos = max_intentos
        self._segundos_bloqueo = segundos_bloqueo
        self._enviador = enviador
        self._minutos_codigo_correo = minutos_codigo_correo
        self._verificador = verificador or VerificadorTotp()
        self._fallos: dict[str, tuple[int, float]] = {}  # usuario -> (fallos, bloqueado_hasta)
        self._ultima_actividad: dict[int, float] = {}
        self._candado = threading.Lock()

    def registrar(
        self,
        usuario: str,
        correo: str,
        contrasena: str,
        generos: list[str],
        *,
        sexo: str | None = None,
        peliculas: list[int] | None = None,
    ) -> VerificacionPendiente:
        """Crea la cuenta sin activar y envía el código de confirmación al correo."""
        correo = correo.strip().lower()
        anterior = self._cuentas.buscar_por_correo(correo)
        if anterior is not None and not anterior.correo_verificado:
            # Nadie demostró ser dueño de ese correo: la cuenta sin confirmar no
            # debe impedir que la persona real se registre con él.
            self._descartar(anterior.user_id)
        cuenta = self._cuentas.crear(usuario, cifrar_contrasena(contrasena), correo)
        # Lo que el usuario dice en el registro es todo lo que se sabe de él: el
        # controlador usa los géneros y las películas para sus primeras
        # recomendaciones. Se guarda en el perfil (SQL) para que esté disponible
        # desde la primera petición, sin esperar a la cola de eventos.
        atributos = {"peliculas_iniciales": ",".join(str(i) for i in peliculas or [])}
        if sexo:
            atributos["sexo"] = sexo
        perfil = UserProfile(
            user_id=cuenta.user_id, preferred_categories=tuple(generos), attributes=atributos
        )
        self._perfiles.save(perfil)
        self._enviar_codigo(cuenta)
        return VerificacionPendiente(correo=correo, minutos=self._minutos_codigo_correo)

    def confirmar_correo(self, correo: str, codigo: str) -> SesionIniciada:
        """Activa la cuenta si el código coincide y no ha vencido."""
        cuenta = self._cuentas.buscar_por_correo(correo.strip().lower())
        if cuenta is None or cuenta.correo_verificado:
            raise CodigoInvalidoError("no account waiting for confirmation")
        self._revisar_bloqueo(cuenta.usuario)
        vigente = (cuenta.codigo_correo_expira or 0) > time.time()
        if not (vigente and hmac.compare_digest(cuenta.codigo_correo or "", huella(codigo))):
            self._anotar_fallo(cuenta.usuario)
            raise CodigoInvalidoError("wrong or expired e-mail code")
        self._fallos.pop(cuenta.usuario, None)
        self._cuentas.marcar_correo_verificado(cuenta.user_id)
        return self._emitir_sesion(cuenta.user_id, cuenta.usuario)

    def reenviar_codigo(self, correo: str) -> None:
        """Manda un código nuevo. No dice si el correo existe, para no revelarlo."""
        cuenta = self._cuentas.buscar_por_correo(correo.strip().lower())
        if cuenta is not None and not cuenta.correo_verificado:
            self._enviar_codigo(cuenta)

    def iniciar_sesion(self, correo: str, contrasena: str) -> SesionIniciada | DobleFactorRequerido:
        # Solo se entra con el correo. El nombre de usuario aparece en pantalla
        # y el correo no, así que para intentar entrar a una cuenta ajena
        # primero hay que conocer su correo.
        correo = correo.strip().lower()
        cuenta = self._cuentas.buscar_por_correo(correo)
        clave = cuenta.usuario if cuenta else correo
        self._revisar_bloqueo(clave)
        guardado = cuenta.hash_contrasena if cuenta else _HASH_DE_RELLENO
        if not contrasena_correcta(contrasena, guardado) or cuenta is None:
            self._anotar_fallo(clave)
            raise CredencialesInvalidasError("invalid email or password")
        self._fallos.pop(clave, None)
        if not cuenta.correo_verificado:
            self._enviar_codigo(cuenta)
            raise CorreoSinVerificarError("e-mail not confirmed yet")
        if cuenta.totp_activo:
            # La contraseña es correcta, pero falta el código: se entrega un
            # token que solo sirve para el segundo paso y dura 5 minutos.
            token = self._firmar(cuenta.user_id, "doble_factor", MINUTOS_PARA_EL_CODIGO)
            return DobleFactorRequerido(token_doble_factor=token)
        return self._emitir_sesion(cuenta.user_id, cuenta.usuario)

    def completar_doble_factor(self, token: str, codigo: str) -> SesionIniciada:
        cuenta = self._cuenta_de(self._leer_token(token, "doble_factor"))
        self._revisar_bloqueo(cuenta.usuario)
        if not self._codigo_correcto(cuenta, codigo):
            self._anotar_fallo(cuenta.usuario)
            raise CodigoInvalidoError("wrong TOTP or backup code")
        self._fallos.pop(cuenta.usuario, None)
        return self._emitir_sesion(cuenta.user_id, cuenta.usuario)

    def iniciar_alta_doble_factor(self, user_id: int) -> ActivacionDobleFactor:
        """Genera el secreto y su QR. No se activa hasta confirmar un código."""
        cuenta = self._cuenta_de(user_id)
        secreto = nuevo_secreto()
        self._cuentas.guardar_doble_factor(user_id, secreto, activo=False, codigos_respaldo="")
        return ActivacionDobleFactor(
            secreto=secreto, qr_svg=qr_para_alta(secreto, cuenta.correo or cuenta.usuario)
        )

    def confirmar_doble_factor(self, user_id: int, codigo: str) -> list[str]:
        """Activa la verificación si el código de la app coincide; devuelve los de respaldo."""
        cuenta = self._cuenta_de(user_id)
        if not cuenta.totp_secreto or not self._verificador.es_valido(
            user_id, cuenta.totp_secreto, codigo
        ):
            raise CodigoInvalidoError("wrong TOTP code while enabling")
        codigos = nuevos_codigos_respaldo()
        huellas = ",".join(huella(c) for c in codigos)
        self._cuentas.guardar_doble_factor(
            user_id, cuenta.totp_secreto, activo=True, codigos_respaldo=huellas
        )
        return codigos

    def desactivar_doble_factor(self, user_id: int, codigo: str) -> None:
        cuenta = self._cuenta_de(user_id)
        if not cuenta.totp_activo or not self._codigo_correcto(cuenta, codigo):
            raise CodigoInvalidoError("wrong code while disabling")
        self._cuentas.guardar_doble_factor(user_id, None, activo=False, codigos_respaldo="")

    def eliminar_cuenta(self, user_id: int, contrasena: str, codigo: str = "") -> None:
        """Derecho de cancelación: borra la cuenta y lo que la persona dijo de sí misma.

        Se pide otra vez la contraseña (y el código, si tiene verificación en dos
        pasos) para que una sesión abierta en otra computadora no baste. Las
        interacciones viejas se quedan con el número de usuario, pero ya no hay
        correo ni nombre que las ligue a una persona.
        """
        cuenta = self._cuenta_de(user_id)
        self._revisar_bloqueo(cuenta.usuario)
        if not contrasena_correcta(contrasena, cuenta.hash_contrasena):
            self._anotar_fallo(cuenta.usuario)
            raise CredencialesInvalidasError("wrong password while deleting the account")
        if cuenta.totp_activo and not self._codigo_correcto(cuenta, codigo):
            self._anotar_fallo(cuenta.usuario)
            raise CodigoInvalidoError("wrong code while deleting the account")
        self._descartar(user_id)
        self._ultima_actividad.pop(user_id, None)

    def usuario_del_token(self, token: str) -> int:
        user_id = self._leer_token(token, "sesion")
        self._ultima_actividad[user_id] = time.time()
        return user_id

    def cuenta(self, user_id: int) -> Cuenta | None:
        return self._cuentas.obtener(user_id)

    def usuarios_activos(self, minutos: int = 5) -> int:
        limite = time.time() - minutos * 60
        return sum(1 for momento in list(self._ultima_actividad.values()) if momento >= limite)

    def _codigo_correcto(self, cuenta: Cuenta, codigo: str) -> bool:
        """Código de la app, o un código de respaldo que se consume al usarlo."""
        if cuenta.totp_secreto and self._verificador.es_valido(
            cuenta.user_id, cuenta.totp_secreto, codigo
        ):
            return True
        guardados = [h for h in cuenta.codigos_respaldo.split(",") if h]
        if huella(codigo) not in guardados:
            return False
        guardados.remove(huella(codigo))
        self._cuentas.guardar_doble_factor(
            cuenta.user_id, cuenta.totp_secreto, cuenta.totp_activo, ",".join(guardados)
        )
        return True

    def _enviar_codigo(self, cuenta: Cuenta) -> None:
        ahora = int(time.time())
        enviado = (cuenta.codigo_correo_expira or 0) - self._minutos_codigo_correo * 60
        if ahora - enviado < SEGUNDOS_ENTRE_CORREOS:
            return  # ya se mandó uno hace menos de un minuto
        codigo = f"{secrets.randbelow(10**6):06d}"
        try:
            self._enviador.enviar(
                cuenta.correo or "",
                f"Tu código de Cartelera Inteligente: {codigo}",
                f"Hola, {cuenta.usuario}.\n\n"
                f"Tu código para confirmar tu correo es: {codigo}\n\n"
                f"Sirve durante {self._minutos_codigo_correo} minutos. Si no creaste una cuenta "
                "en Cartelera Inteligente, ignora este mensaje.\n",
            )
        except Exception as exc:  # noqa: BLE001 - SMTP caído, credenciales mal, sin red
            raise CorreoNoEnviadoError(f"{type(exc).__name__}: {exc}") from exc
        # Se guarda solo si el correo salió, para poder reintentar enseguida.
        expira = ahora + self._minutos_codigo_correo * 60
        self._cuentas.guardar_codigo_correo(cuenta.user_id, huella(codigo), expira)

    def _descartar(self, user_id: int) -> None:
        """Borra una cuenta sin confirmar junto con los datos de su perfil."""
        self._cuentas.eliminar(user_id)
        self._perfiles.save(UserProfile(user_id=user_id, preferred_categories=(), attributes={}))

    def _cuenta_de(self, user_id: int) -> Cuenta:
        cuenta = self._cuentas.obtener(user_id)
        if cuenta is None:
            raise SesionInvalidaError(f"user {user_id} has no account")
        return cuenta

    def _emitir_sesion(self, user_id: int, usuario: str) -> SesionIniciada:
        token = self._firmar(user_id, "sesion", self._minutos_sesion)
        self._ultima_actividad[user_id] = time.time()
        return SesionIniciada(
            token=token, user_id=user_id, usuario=usuario, minutos_sesion=self._minutos_sesion
        )

    def _firmar(self, user_id: int, tipo: str, minutos: int) -> str:
        ahora = datetime.now(UTC)
        vence = ahora + timedelta(minutes=minutos)
        datos = {"sub": str(user_id), "typ": tipo, "iat": ahora, "exp": vence}
        return jwt.encode(datos, self._secreto, algorithm="HS256")

    def _leer_token(self, token: str, tipo: str) -> int:
        """El campo ``typ`` impide usar el token del segundo paso como sesión."""
        try:
            datos = jwt.decode(token, self._secreto, algorithms=["HS256"])
        except jwt.PyJWTError as exc:
            raise SesionInvalidaError("invalid or expired token", detail=str(exc)) from exc
        if datos.get("typ") != tipo:
            raise SesionInvalidaError(f"expected a {tipo} token")
        return int(datos["sub"])

    def _revisar_bloqueo(self, usuario: str) -> None:
        _, bloqueado_hasta = self._fallos.get(usuario, (0, 0.0))
        if time.time() < bloqueado_hasta:
            raise DemasiadosIntentosError(f"user {usuario!r} is temporarily locked")

    def _anotar_fallo(self, usuario: str) -> None:
        with self._candado:
            fallos, _ = self._fallos.get(usuario, (0, 0.0))
            fallos += 1
            if fallos >= self._max_intentos:
                self._fallos[usuario] = (0, time.time() + self._segundos_bloqueo)
            else:
                self._fallos[usuario] = (fallos, 0.0)


__all__ = ["ServicioAutenticacion", "cifrar_contrasena", "contrasena_correcta"]
