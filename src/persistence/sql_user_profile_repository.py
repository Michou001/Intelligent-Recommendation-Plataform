"""SqlUserProfileRepository - relational adapter for FR-01 (Section 3.2).

Implements ``IUserProfileRepository`` over SQLAlchemy. The engine URL comes
from configuration, so the same class serves SQLite in development and
PostgreSQL in production without a code change: that is the whole extent of
what DIP buys here, as Section 3.2 is careful to state.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import Settings, load_settings
from persistence import get_engine, user_profiles
from persistence.interfaces import IUserProfileRepository, UserId, UserProfile

#: Separator used to store the category list in a single text column, so the
#: schema stays portable across SQLite and PostgreSQL.
CATEGORY_SEPARATOR = "|"


class SqlUserProfileRepository(IUserProfileRepository):
    """Relational storage of user profiles and explicit preferences."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SqlUserProfileRepository:
        """Build the adapter against the configured SQL endpoint."""
        settings = settings or load_settings()
        return cls(get_engine(settings.persistence.sql_url))

    def get_by_id(self, user_id: UserId) -> UserProfile | None:
        statement = select(user_profiles).where(user_profiles.c.user_id == int(user_id))
        with self._engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        return None if row is None else _to_profile(dict(row))

    def exists(self, user_id: UserId) -> bool:
        statement = (
            select(func.count())
            .select_from(user_profiles)
            .where(user_profiles.c.user_id == int(user_id))
        )
        with self._engine.connect() as connection:
            return bool(connection.execute(statement).scalar_one())

    def save(self, profile: UserProfile) -> None:
        values = {
            "user_id": int(profile.user_id),
            "preferred_categories": CATEGORY_SEPARATOR.join(profile.preferred_categories),
            "attributes": json.dumps(dict(profile.attributes)),
        }
        with self._engine.begin() as connection:
            if connection.dialect.name == "sqlite":
                statement = sqlite_insert(user_profiles).values(**values)
                statement = statement.on_conflict_do_update(
                    index_elements=[user_profiles.c.user_id],
                    set_={
                        "preferred_categories": statement.excluded.preferred_categories,
                        "attributes": statement.excluded.attributes,
                    },
                )
                connection.execute(statement)
            else:
                # Portable read-then-write for engines without a SQLite-style
                # upsert; the two adapters keep the same observable behaviour.
                exists = connection.execute(
                    select(func.count())
                    .select_from(user_profiles)
                    .where(user_profiles.c.user_id == values["user_id"])
                ).scalar_one()
                if exists:
                    connection.execute(
                        user_profiles.update()
                        .where(user_profiles.c.user_id == values["user_id"])
                        .values(**values)
                    )
                else:
                    connection.execute(user_profiles.insert().values(**values))

    def count(self) -> int:
        """Number of stored profiles, used by the seeding CLI and /health."""
        with self._engine.connect() as connection:
            return int(
                connection.execute(select(func.count()).select_from(user_profiles)).scalar_one()
            )


def _to_profile(row: dict[str, Any]) -> UserProfile:
    raw_categories = str(row.get("preferred_categories") or "")
    categories = tuple(c for c in raw_categories.split(CATEGORY_SEPARATOR) if c)
    try:
        attributes = json.loads(str(row.get("attributes") or "{}"))
    except ValueError:
        attributes = {}
    return UserProfile(
        user_id=int(row["user_id"]),
        preferred_categories=categories,
        attributes={str(k): str(v) for k, v in attributes.items()},
    )


__all__ = ["SqlUserProfileRepository"]
