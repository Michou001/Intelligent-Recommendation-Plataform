"""Process entry point of the API container (Milestone 3, scalability).

    python -m application.server            # reads HOST, PORT, WORKERS, LOG_LEVEL

It runs the same ASGI application as ``uvicorn application.api:app``. It
exists because of a bottleneck measured during the Milestone 3 load test: with
``--workers N > 1`` on Linux, every keep-alive request took a constant ~48 ms,
even ``/health``, while a single worker answered in ~1.5 ms.

Cause: in multi-worker mode uvicorn binds the shared listening socket itself
with ``socket.socket(family, type)``, whose ``proto`` is 0. asyncio enables
``TCP_NODELAY`` on accepted connections only when ``proto == IPPROTO_TCP``, so
Nagle's algorithm stays on and collides with the client's delayed ACK (~40 ms).
On Linux ``TCP_NODELAY`` set on a listening socket is inherited by the sockets
it accepts, so setting it once here removes the delay for every worker.
"""

from __future__ import annotations

import os
import secrets
import socket

import uvicorn
from uvicorn.supervisors import Multiprocess

APP = "application.api:app"


def build_config() -> uvicorn.Config:
    """Server configuration taken from the environment (Dockerfile, ConfigMap)."""
    return uvicorn.Config(
        APP,
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - container entry point
        port=int(os.environ.get("PORT", "8000")),
        workers=max(1, int(os.environ.get("WORKERS", "1"))),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


def bind_listening_socket(config: uvicorn.Config) -> socket.socket:
    """Bind the shared socket with Nagle disabled for every accepted connection."""
    sock = config.bind_socket()
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def ip_en_la_red() -> str:
    """IP de esta máquina en la red local, la que deben abrir las otras máquinas.

    Conectar un socket UDP no envía nada; solo hace que el sistema elija la
    interfaz de red que usaría para salir, y de ella se lee la IP.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sonda:
        try:
            sonda.connect(("8.8.8.8", 80))
            return str(sonda.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def main() -> None:
    config = build_config()
    if config.host == "0.0.0.0":  # noqa: S104 - se compara, no se abre
        print(f"\n  Desde otras máquinas abre: http://{ip_en_la_red()}:{config.port}/ui/\n")
    if config.workers > 1:
        # Cada proceso firma y revisa tokens de sesión. Si cada uno generara su
        # propio secreto, un token firmado por un proceso fallaría en otro. Se
        # genera uno aquí y los procesos hijos lo heredan por el entorno.
        os.environ.setdefault("RECO_JWT_SECRETO", secrets.token_urlsafe(32))
        Multiprocess(config, sockets=[bind_listening_socket(config)]).run()
    else:
        # A single worker lets asyncio create the socket, which already
        # enables TCP_NODELAY on each connection.
        uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
