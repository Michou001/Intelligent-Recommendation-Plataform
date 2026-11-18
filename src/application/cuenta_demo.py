"""Crea una cuenta ligada a un usuario de MovieLens, para mostrar la fase personalizada.

    python -m application.cuenta_demo --usuario demo --correo demo@correo.com --id 1

Los usuarios de MovieLens no tienen contraseña ni correo; con esta cuenta se
puede entrar como uno de ellos y ver recomendaciones de filtrado colaborativo
(ALS) junto a las de un usuario recién registrado. Se inicia sesión con el
correo y la contraseña se pide sin mostrarla.
"""

from __future__ import annotations

import argparse
import getpass

from application.autenticacion import cifrar_contrasena
from presentation.schemas import validar_contrasena


def main(argv: list[str] | None = None) -> int:
    from config.settings import load_settings
    from persistence.sql_repositorio_cuentas import SqlRepositorioCuentas
    from persistence.sql_user_profile_repository import SqlUserProfileRepository

    parser = argparse.ArgumentParser(prog="python -m application.cuenta_demo")
    parser.add_argument("--usuario", required=True)
    parser.add_argument("--id", type=int, required=True, help="user_id de MovieLens (1 a 610)")
    parser.add_argument("--correo", required=True, help="con este correo se inicia sesión")
    args = parser.parse_args(argv)

    settings = load_settings()
    if not SqlUserProfileRepository.from_settings(settings).exists(args.id):
        print(f"No existe el usuario {args.id} en la base de datos.")
        return 1
    contrasena = getpass.getpass("Contraseña (8+ caracteres, mayúscula, número y símbolo): ")
    try:
        validar_contrasena(contrasena)
    except ValueError as error:
        print(f"La contraseña {error}.")
        return 1
    SqlRepositorioCuentas.from_settings(settings).crear(
        args.usuario,
        cifrar_contrasena(contrasena),
        correo=args.correo.strip().lower(),
        correo_verificado=True,  # la crea quien administra el servidor
        user_id=args.id,
    )
    print(f"Cuenta '{args.usuario}' creada para el usuario {args.id} de MovieLens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
