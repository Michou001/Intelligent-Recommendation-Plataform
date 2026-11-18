"""Verificación en dos pasos con TOTP (RFC 6238): la de Google o Microsoft Authenticator.

El servidor y la app del celular comparten un secreto. Cada 30 segundos los dos
calculan el mismo código de 6 dígitos a partir del secreto y la hora, así que
el código demuestra que la persona tiene su celular, sin enviar SMS ni correos.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable

import pyotp
import segno

EMISOR = "Cartelera Inteligente"
PASO_SEGUNDOS = 30
NUM_CODIGOS_RESPALDO = 8


def nuevo_secreto() -> str:
    return pyotp.random_base32()


def qr_para_alta(secreto: str, usuario: str) -> str:
    """SVG del código QR que se escanea con la app autenticadora."""
    enlace = pyotp.TOTP(secreto).provisioning_uri(name=usuario, issuer_name=EMISOR)
    return str(segno.make(enlace, error="m").svg_inline(scale=5))


def nuevos_codigos_respaldo() -> list[str]:
    """Códigos de un solo uso por si la persona pierde su celular (p. ej. 'a1b2-c3d4')."""
    return [f"{secrets.token_hex(2)}-{secrets.token_hex(2)}" for _ in range(NUM_CODIGOS_RESPALDO)]


def huella(codigo: str) -> str:
    """Los códigos de respaldo se guardan como hash, igual que una contraseña."""
    return hashlib.sha256(codigo.strip().lower().encode()).hexdigest()


class VerificadorTotp:
    """Comprueba códigos TOTP y no acepta dos veces el mismo código.

    Sin esa regla, alguien que viera el código en pantalla podría usarlo
    durante los 30 segundos que sigue siendo válido.
    """

    def __init__(self, reloj: Callable[[], float] = time.time) -> None:
        self._reloj = reloj
        self._ultimo_paso: dict[int, int] = {}
        self._candado = threading.Lock()

    def es_valido(self, user_id: int, secreto: str, codigo: str) -> bool:
        totp = pyotp.TOTP(secreto)
        ahora = self._reloj()
        # Se acepta el código anterior y el siguiente por si el reloj del
        # celular está unos segundos adelantado o atrasado.
        for desfase in (-1, 0, 1):
            momento = int(ahora) + desfase * PASO_SEGUNDOS
            paso = momento // PASO_SEGUNDOS
            if hmac.compare_digest(totp.at(momento), codigo.strip()):
                with self._candado:
                    if paso <= self._ultimo_paso.get(user_id, -1):
                        return False
                    self._ultimo_paso[user_id] = paso
                return True
        return False


__all__ = [
    "VerificadorTotp",
    "huella",
    "nuevo_secreto",
    "nuevos_codigos_respaldo",
    "qr_para_alta",
]
