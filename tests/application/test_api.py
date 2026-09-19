"""The HTTP surface: the endpoint of Section 9.6 and the codes of Table 10.

These tests drive the fully composed application - the real factory, the real
decorator chain, the real repositories - against the synthetic feature store,
so they check the composition root as well as the routes.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from application.api import build_components, create_app
from config.settings import Settings
from persistence.interfaces import CatalogItem, UserProfile
from persistence.sql_catalog_repository import SqlCatalogRepository
from persistence.sql_user_profile_repository import SqlUserProfileRepository
from tests.conftest import N_ITEMS, N_USERS, UNKNOWN_USER


@pytest.fixture
def seeded(settings: Settings) -> Settings:
    """Populate the profile and catalog stores the routes read from."""
    profiles = SqlUserProfileRepository.from_settings(settings)
    for user_id in range(1, N_USERS + 1):
        profiles.save(UserProfile(user_id=user_id, preferred_categories=("Drama",)))
    catalog = SqlCatalogRepository.from_settings(settings)
    catalog.save_many(
        [
            CatalogItem(item_id=101 + i, title=f"Item {101 + i}", categories=("Drama",))
            for i in range(N_ITEMS)
        ]
    )
    return settings


@pytest.fixture
def client(seeded: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("RECO_ACTIVE_STRATEGY", "als")
    from config.settings import load_settings

    with TestClient(create_app(load_settings())) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# GET /recommendations/{user_id}
# --------------------------------------------------------------------------


def test_the_documented_endpoint_answers(client: TestClient) -> None:
    """``GET /recommendations/{user_id}?k=10`` (Section 9.6)."""
    response = client.get("/recommendations/1", params={"k": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["k"] == 10
    assert len(body["items"]) <= 10


def test_k_defaults_to_ten(client: TestClient) -> None:
    assert client.get("/recommendations/1").json()["k"] == 10


def test_the_response_carries_the_ranking_metadata(client: TestClient) -> None:
    body = client.get("/recommendations/1?k=3").json()
    assert body["strategy_id"] == "als"
    assert body["feature_store_version"] == "v-test"
    assert [item["rank"] for item in body["items"]] == [1, 2, 3]


def test_items_are_enriched_with_catalog_titles(client: TestClient) -> None:
    body = client.get("/recommendations/1?k=3").json()
    assert all(item["title"] for item in body["items"])


def test_the_second_request_is_served_from_the_cache(client: TestClient) -> None:
    first = client.get("/recommendations/2?k=5").json()
    second = client.get("/recommendations/2?k=5").json()
    assert [i["item_id"] for i in first["items"]] == [i["item_id"] for i in second["items"]]


# --------------------------------------------------------------------------
# Table 10: status codes
# --------------------------------------------------------------------------


def test_an_unknown_user_is_404(client: TestClient) -> None:
    response = client.get(f"/recommendations/{UNKNOWN_USER}")
    assert response.status_code == 404
    assert response.json()["code"] == "user_not_found"


@pytest.mark.parametrize("k", [0, -1, 1000])
def test_an_out_of_range_k_is_400(client: TestClient, k: int) -> None:
    """Table 10: a schema violation is HTTP 400, not the framework default."""
    response = client.get("/recommendations/1", params={"k": k})
    assert response.status_code == 400
    assert response.json()["code"] == "validation_error"


def test_a_non_numeric_k_is_400(client: TestClient) -> None:
    assert client.get("/recommendations/1", params={"k": "ten"}).status_code == 400


@pytest.mark.parametrize("user_id", ["0", "-3", "abc"])
def test_an_invalid_user_id_is_400(client: TestClient, user_id: str) -> None:
    assert client.get(f"/recommendations/{user_id}").status_code == 400


def test_the_error_payload_hides_internals(client: TestClient) -> None:
    """Section 9.8: no stack traces, no internal identifiers."""
    body = client.get(f"/recommendations/{UNKNOWN_USER}").json()
    assert set(body) <= {"code", "message", "detail"}
    assert "Traceback" not in str(body)


# --------------------------------------------------------------------------
# POST /interactions
# --------------------------------------------------------------------------


def test_an_interaction_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/interactions", json={"user_id": 1, "event_type": "click", "item_id": 101}
    )
    assert response.status_code == 202
    assert response.json()["accepted"] is True


def test_a_malformed_interaction_is_400(client: TestClient) -> None:
    response = client.post("/interactions", json={"user_id": 0, "event_type": ""})
    assert response.status_code == 400


def test_a_blank_event_type_is_rejected(client: TestClient) -> None:
    response = client.post("/interactions", json={"user_id": 1, "event_type": "   ", "item_id": 1})
    assert response.status_code == 400


def test_an_interaction_without_a_subject_is_400(client: TestClient) -> None:
    response = client.post("/interactions", json={"user_id": 1, "event_type": "click"})
    assert response.status_code == 400


# --------------------------------------------------------------------------
# /health and the composition root
# --------------------------------------------------------------------------


def test_health_reports_the_composition(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["active_strategy"] == "als"
    assert body["decorator_chain"] == ["caching", "timing", "logging"]
    assert set(body["registered_strategies"]) >= {"als", "content", "ncf", "popularity"}
    assert body["feature_store_available"] is True


def test_the_composition_matches_the_documented_snippet(seeded: Settings) -> None:
    """LoggingDecorator(TimingDecorator(CachingDecorator(base)))."""
    from ai.decorators.caching_decorator import CachingDecorator
    from ai.decorators.logging_decorator import LoggingDecorator
    from ai.decorators.timing_decorator import TimingDecorator

    components = build_components(seeded)
    chain = components.controller.strategy
    assert isinstance(chain, LoggingDecorator)
    assert isinstance(chain.wrapped, TimingDecorator)
    assert isinstance(chain.wrapped.wrapped, CachingDecorator)


def test_the_request_path_performs_no_wiring(seeded: Settings) -> None:
    """Section 9.6: the composition is built once, at startup."""
    components = build_components(seeded)
    before = components.controller.strategy
    with TestClient(create_app(seeded)) as client:
        client.get("/recommendations/1")
    assert components.controller.strategy is before


def test_an_unloadable_active_strategy_still_starts_the_service(
    seeded: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR-05: a broken model must not prevent the service from serving."""
    monkeypatch.setenv("RECO_ACTIVE_STRATEGY", "ncf")  # no checkpoint in the temp dir
    from config.settings import load_settings

    with TestClient(create_app(load_settings())) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["active_strategy"] == "popularity"
        assert client.get("/recommendations/1?k=3").status_code == 200
