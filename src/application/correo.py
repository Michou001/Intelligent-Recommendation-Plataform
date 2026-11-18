"""Envío de correos: el código que demuestra que el correo es de quien se registra.

Sin este paso, cualquiera podría registrarse con el correo de otra persona.
Hay dos formas de enviar:

- ``EnviadorSmtp``: el envío real, por SMTP con TLS (por ejemplo, una cuenta de
  Gmail con contraseña de aplicación). Se activa al definir RECO_SMTP_USUARIO.
- ``BuzonLocal``: para desarrollo y pruebas, sin servidor de correo. No envía
  nada: anota el mensaje en un archivo y lo muestra en la consola del servidor.
"""

from __future__ import annotations

import smtplib
import ssl
from abc import ABC, abstractmethod
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from config.settings import Settings


class IEnviadorCorreo(ABC):
    @abstractmethod
    def enviar(self, destinatario: str, asunto: str, texto: str) -> None:
        """Envía el mensaje o lanza una excepción si no pudo."""


class EnviadorSmtp(IEnviadorCorreo):
    def __init__(self, host: str, puerto: int, usuario: str, contrasena: str, remitente: str):
        self._host = host
        self._puerto = puerto
        self._usuario = usuario
        self._contrasena = contrasena
        self._remitente = remitente or usuario

    def enviar(self, destinatario: str, asunto: str, texto: str) -> None:
        mensaje = EmailMessage()
        mensaje["From"] = f"Cartelera Inteligente <{self._remitente}>"
        mensaje["To"] = destinatario
        mensaje["Subject"] = asunto
        mensaje.set_content(texto)
        with smtplib.SMTP(self._host, self._puerto, timeout=15) as servidor:
            servidor.starttls(context=ssl.create_default_context())
            servidor.login(self._usuario, self._contrasena)
            servidor.send_message(mensaje)


class BuzonLocal(IEnviadorCorreo):
    def __init__(self, archivo: Path) -> None:
        self._archivo = archivo

    def enviar(self, destinatario: str, asunto: str, texto: str) -> None:
        registro = (
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Para: {destinatario} | {asunto}\n{texto}\n\n"
        )
        self._archivo.parent.mkdir(parents=True, exist_ok=True)
        with self._archivo.open("a", encoding="utf-8") as salida:
            salida.write(registro)
        print(f"\n  [Correo sin enviar, modo local] {registro}", flush=True)


def enviador_desde(settings: Settings) -> IEnviadorCorreo:
    web = settings.web
    if web.smtp_usuario:
        return EnviadorSmtp(
            web.smtp_host,
            web.smtp_puerto,
            web.smtp_usuario,
            web.smtp_contrasena,
            web.smtp_remitente,
        )
    return BuzonLocal(web.buzon_correos)


__all__ = ["BuzonLocal", "EnviadorSmtp", "IEnviadorCorreo", "enviador_desde"]
