"""RecommendationController (FR-01, FR-04, FR-05, NFR-04, NFR-05).

Every collaborator here is a double. That is the concrete payoff Section 7
claims for the architecture: the controller can be tested without a model, a
broker, or a database, because it depends only on abstractions.
"""

from __future__ import annotations

import inspect

import pytest

from ai.interfaces import UserFeatures
from application.recommendation_controller import RecommendationController
from config.settings import Settings
from data.feature_store_manager import FeatureStoreManager, ParquetFeatureStoreManager
from errors import (
    ColdStartError,
    FeatureStoreUnavailableError,
    UserNotFoundError,
)
from messaging.interaction_event_queue import InMemoryInteractionEventQueue
from persistence.interfaces import (
    CatalogItem,
    ICatalogRepository,
    IUserProfileRepository,
    UserProfile,
)
from tests.conftest import ColdStartStrategy, FailingStrategy, StubStrategy


class StubProfiles(IUserProfileRepository):
    def __init__(self, known: set[int] | None = None, *, broken: bool = False) -> None:
        self._known = known if known is not None else {1, 2, 3}
        self._broken = broken

    def get_by_id(self, user_id: int) -> UserProfile | None:
        return UserProfile(user_id=user_id) if user_id in self._known else None

    def exists(self, user_id: int) -> bool:
        if self._broken:
            raise ConnectionError("profile store is down")
        return user_id in self._known

    def save(self, profile: UserProfile) -> None:
        self._known.add(profile.user_id)


class StubCatalog(ICatalogRepository):
    def __init__(self, *, broken: bool = False) -> None:
        self._broken = broken

    def get_item(self, item_id: int) -> CatalogItem | None:
        return CatalogItem(item_id=item_id, title=f"Item {item_id}")

    def get_items(self, item_ids) -> list[CatalogItem]:  # noqa: ANN001
        if self._broken:
            raise ConnectionError("catalog is down")
        return [
            CatalogItem(item_id=int(i), title=f"Item {i}", categories=("Drama",)) for i in item_ids
        ]

    def list_by_category(self, category: str, limit: int = 100) -> list[CatalogItem]:
        return []

    def count(self) -> int:
        return 0

    def save(self, item: CatalogItem) -> None:
        return None


class BrokenFeatureStore(FeatureStoreManager):
    def get_user_features(self, user_id: int) -> UserFeatures:
        raise FeatureStoreUnavailableError("injected feature store outage")

    def is_available(self) -> bool:
        return False

    @property
    def version(self) -> str:
        return "v-broken"


class ColdFeatureStore(FeatureStoreManager):
    def get_user_features(self, user_id: int) -> UserFeatures:
        raise ColdStartError("no history")

    def is_available(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return "v-test"


@pytest.fixture
def feature_store(settings: Settings) -> ParquetFeatureStoreManager:
    return ParquetFeatureStoreManager.from_settings(settings)


def build_controller(
    settings: Settings,
    *,
    strategy=None,  # noqa: ANN001
    feature_store=None,  # noqa: ANN001
    fallback=None,  # noqa: ANN001
    queue=None,  # noqa: ANN001
    profiles=None,  # noqa: ANN001
    catalog=None,  # noqa: ANN001
) -> RecommendationController:
    return RecommendationController(
        strategy=strategy or StubStrategy(),
        feature_store=feature_store or ParquetFeatureStoreManager.from_settings(settings),
        event_queue=queue,
        catalog_repository=catalog if catalog is not None else StubCatalog(),
        profile_repository=profiles if profiles is not None else StubProfiles({1, 2, 3, 99}),
        fallback_strategy=fallback or StubStrategy([901, 902, 903], model_version="pop-v1"),
        settings=settings,
    )


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_it_serves_a_ranking(settings: Settings) -> None:
    response = build_controller(settings).recommend(1, 3)
    assert response.user_id == 1
    assert len(response.items) == 3
    assert response.degraded is False
    assert response.degradation_reason is None


def test_the_ranking_is_positional(settings: Settings) -> None:
    response = build_controller(settings).recommend(1, 3)
    assert [item.rank for item in response.items] == [1, 2, 3]


def test_the_ranking_is_enriched_from_the_catalog(settings: Settings) -> None:
    response = build_controller(settings).recommend(1, 2)
    assert all(item.title for item in response.items)
    assert response.items[0].categories == ["Drama"]


def test_it_reports_the_strategy_and_versions(settings: Settings) -> None:
    response = build_controller(settings).recommend(1, 2)
    assert response.strategy_id == "stub"
    assert response.model_version == "stub-v1"
    assert response.feature_store_version == "v-test"


def test_k_defaults_to_the_configured_value(settings: Settings) -> None:
    response = build_controller(settings, strategy=StubStrategy(list(range(200, 220)))).recommend(1)
    assert response.k == settings.model.default_k


def test_it_measures_its_own_latency(settings: Settings) -> None:
    assert build_controller(settings).recommend(1, 3).latency_ms >= 0


# --------------------------------------------------------------------------
# Fail Fast (NFR-04)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("user_id", [0, -1, -999])
def test_an_invalid_user_id_is_rejected_before_anything_runs(
    settings: Settings, user_id: int
) -> None:
    strategy = StubStrategy()
    with pytest.raises(ValueError, match="user_id"):
        build_controller(settings, strategy=strategy).recommend(user_id, 3)
    assert strategy.calls == 0


@pytest.mark.parametrize("k", [0, -5])
def test_an_invalid_k_is_rejected(settings: Settings, k: int) -> None:
    with pytest.raises(ValueError, match="k must"):
        build_controller(settings).recommend(1, k)


def test_a_k_above_the_maximum_is_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        build_controller(settings).recommend(1, settings.model.max_k + 1)


def test_the_ai_layer_is_never_invoked_on_an_invalid_request(settings: Settings) -> None:
    """NFR-04: reject before wasting compute."""
    strategy = StubStrategy()
    with pytest.raises(ValueError):
        build_controller(settings, strategy=strategy).recommend(-1, 3)
    assert strategy.calls == 0


# --------------------------------------------------------------------------
# Table 10 behaviours
# --------------------------------------------------------------------------


def test_an_unknown_user_is_a_404(settings: Settings) -> None:
    with pytest.raises(UserNotFoundError) as excinfo:
        build_controller(settings, profiles=StubProfiles({1})).recommend(4242, 3)
    assert excinfo.value.status_code == 404


def test_no_inference_is_attempted_for_an_unknown_user(settings: Settings) -> None:
    strategy = StubStrategy()
    with pytest.raises(UserNotFoundError):
        build_controller(settings, strategy=strategy, profiles=StubProfiles({1})).recommend(4242, 3)
    assert strategy.calls == 0


def test_a_cold_start_falls_back_to_popularity(settings: Settings) -> None:
    """FR-05: the fallback answers, the request still succeeds."""
    controller = build_controller(settings, feature_store=ColdFeatureStore())
    response = controller.recommend(1, 3)
    assert response.degraded is True
    assert response.degradation_reason == ColdStartError.code
    assert [item.item_id for item in response.items] == [901, 902, 903]


def test_an_unavailable_feature_store_degrades(settings: Settings) -> None:
    controller = build_controller(settings, feature_store=BrokenFeatureStore())
    response = controller.recommend(1, 3)
    assert response.degraded is True
    assert response.degradation_reason == FeatureStoreUnavailableError.code
    assert len(response.items) == 3


def test_a_failing_model_degrades(settings: Settings) -> None:
    """NFR-05: catalog navigation survives a failure of the model."""
    response = build_controller(settings, strategy=FailingStrategy()).recommend(1, 3)
    assert response.degraded is True
    assert response.degradation_reason == "model_unavailable"
    # The fallback served it: its own items and its own model version.
    assert response.model_version == "pop-v1"
    assert [item.item_id for item in response.items] == [901, 902, 903]


def test_an_unexpected_model_exception_also_degrades(settings: Settings) -> None:
    response = build_controller(
        settings, strategy=FailingStrategy(RuntimeError("segfault-ish"))
    ).recommend(1, 3)
    assert response.degraded is True
    assert len(response.items) == 3


def test_a_cold_start_raised_by_the_strategy_degrades(settings: Settings) -> None:
    response = build_controller(settings, strategy=ColdStartStrategy()).recommend(1, 3)
    assert response.degradation_reason == ColdStartError.code


def test_a_publication_failure_does_not_affect_the_response(settings: Settings) -> None:
    """Table 10: EventPublishError does not affect the response."""
    from tests.messaging.test_messaging import BrokenQueue

    response = build_controller(settings, queue=BrokenQueue()).recommend(1, 3)
    assert len(response.items) == 3
    assert response.degraded is False


def test_a_broken_catalog_still_serves_a_ranking(settings: Settings) -> None:
    response = build_controller(settings, catalog=StubCatalog(broken=True)).recommend(1, 3)
    assert len(response.items) == 3
    assert response.items[0].title is None


def test_a_broken_profile_store_does_not_block_the_request(settings: Settings) -> None:
    response = build_controller(settings, profiles=StubProfiles(broken=True)).recommend(1, 3)
    assert len(response.items) == 3


def test_without_a_fallback_a_model_failure_is_not_silently_empty(settings: Settings) -> None:
    from errors import ModelUnavailableError

    controller = RecommendationController(
        strategy=FailingStrategy(),
        feature_store=ParquetFeatureStoreManager.from_settings(settings),
        profile_repository=StubProfiles(),
        fallback_strategy=None,
        settings=settings,
    )
    with pytest.raises(ModelUnavailableError):
        controller.recommend(1, 3)


# --------------------------------------------------------------------------
# The asynchronous branch (FR-06)
# --------------------------------------------------------------------------


def test_serving_publishes_an_interaction_event(settings: Settings) -> None:
    queue = InMemoryInteractionEventQueue()
    response = build_controller(settings, queue=queue).recommend(1, 3)
    assert queue.pending() == 1

    consumed: list = []
    queue.consume(lambda event: (consumed.append(event), True)[1], max_events=1)
    event = consumed[0]
    assert event.user_id == 1
    assert event.recommended_items == tuple(item.item_id for item in response.items)
    assert event.metadata["strategy_id"] == "stub"


def test_the_event_records_whether_the_response_was_degraded(settings: Settings) -> None:
    queue = InMemoryInteractionEventQueue()
    build_controller(settings, queue=queue, feature_store=BrokenFeatureStore()).recommend(1, 3)
    consumed: list = []
    queue.consume(lambda event: (consumed.append(event), True)[1], max_events=1)
    assert consumed[0].metadata["degraded"] == "true"


# --------------------------------------------------------------------------
# Structural claims
# --------------------------------------------------------------------------


def test_the_controller_contains_no_algorithmic_logic() -> None:
    """Section 3.1: it orchestrates, it does not compute a ranking."""
    source = inspect.getsource(RecommendationController)
    for forbidden in ("argpartition", "np.dot", "cosine", "@ ", "argsort"):
        assert forbidden not in source, f"the controller appears to score: {forbidden!r}"


def test_the_controller_names_no_concrete_strategy() -> None:
    source = inspect.getsource(RecommendationController)
    for forbidden in ("PopularityStrategy", "CollaborativeFilteringStrategy", "ModelFactory"):
        assert forbidden not in source
