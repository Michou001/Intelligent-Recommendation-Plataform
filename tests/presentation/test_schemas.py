"""Presentation contracts (NFR-04, Table 7).

Fail Fast is claimed to be a framework feature here rather than hand-written
code. These tests check that the schemas actually reject what the document
says they must reject.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from presentation.schemas import (
    MAX_K,
    ErrorResponse,
    InteractionRequest,
    RecommendationResponse,
    RecommendedItem,
)


def test_a_valid_interaction_is_accepted() -> None:
    request = InteractionRequest(user_id=1, event_type="click", item_id=101)
    assert request.user_id == 1
    assert request.metadata == {}


@pytest.mark.parametrize("user_id", [0, -1, -999])
def test_a_non_positive_user_id_is_rejected(user_id: int) -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(user_id=user_id, event_type="click", item_id=1)


def test_a_non_numeric_user_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(user_id="abc", event_type="click", item_id=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("event_type", ["", "   "])
def test_a_blank_event_type_is_rejected(event_type: str) -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(user_id=1, event_type=event_type, item_id=1)


def test_the_event_type_is_stripped() -> None:
    assert InteractionRequest(user_id=1, event_type="  click ", item_id=1).event_type == "click"


def test_a_negative_item_identifier_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(user_id=1, event_type="click", item_id=-3)


def test_negative_recommended_items_are_rejected() -> None:
    with pytest.raises(ValidationError, match="positive identifiers"):
        InteractionRequest(user_id=1, event_type="click", recommended_items=[1, -2])


def test_an_oversized_recommendation_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InteractionRequest(
            user_id=1, event_type="click", recommended_items=list(range(1, MAX_K + 10))
        )


def test_the_response_is_serialisable() -> None:
    response = RecommendationResponse(
        user_id=1,
        k=2,
        items=[RecommendedItem(item_id=101, rank=1, score=0.9, title="A", categories=["Drama"])],
        strategy_id="als",
        model_version="v1",
        feature_store_version="v1",
    )
    payload = response.model_dump()
    assert payload["degraded"] is False
    assert payload["items"][0]["title"] == "A"


def test_a_degraded_response_carries_its_reason() -> None:
    response = RecommendationResponse(
        user_id=1,
        k=0,
        items=[],
        strategy_id="popularity",
        model_version="v1",
        feature_store_version="v1",
        degraded=True,
        degradation_reason="feature_store_unavailable",
    )
    assert response.degraded is True
    assert response.degradation_reason == "feature_store_unavailable"


def test_a_rank_must_start_at_one() -> None:
    with pytest.raises(ValidationError):
        RecommendedItem(item_id=1, rank=0, score=0.5)


def test_the_error_payload_has_a_stable_shape() -> None:
    payload = ErrorResponse(code="cold_start", message="no history").model_dump()
    assert set(payload) == {"code", "message", "detail"}
