"""The six exception types of Table 10 (Section 9.8).

The table specifies, for each type, when it is raised, the HTTP response, and
the recovery behaviour. This module pins the first two; the recovery
behaviour is exercised in ``test_failure_injection.py`` and in the controller
tests.
"""

from __future__ import annotations

import pytest

from errors import (
    DEGRADATION_REASONS,
    ColdStartError,
    EventPublishError,
    FeatureStoreUnavailableError,
    ModelUnavailableError,
    RecommendationPlatformError,
    UserNotFoundError,
)

#: Table 10, one row per exception: (type, status code, degradable).
TABLE_10 = [
    (UserNotFoundError, 404, False),
    (ColdStartError, 200, True),
    (FeatureStoreUnavailableError, 200, True),
    (ModelUnavailableError, 200, True),
    (EventPublishError, 200, True),
]


@pytest.mark.parametrize(("error_type", "status", "degradable"), TABLE_10)
def test_each_type_matches_its_row(
    error_type: type[RecommendationPlatformError], status: int, degradable: bool
) -> None:
    error = error_type("something happened")
    assert error.status_code == status
    assert error.degradable is degradable


def test_every_type_shares_one_base() -> None:
    """A single handler translates the whole vocabulary (Section 9.8)."""
    for error_type, _status, _degradable in TABLE_10:
        assert issubclass(error_type, RecommendationPlatformError)


def test_the_codes_are_distinct() -> None:
    codes = [error_type.code for error_type, _s, _d in TABLE_10]
    assert len(set(codes)) == len(codes)


def test_the_payload_never_carries_a_stack_trace() -> None:
    payload = ModelUnavailableError("boom", detail="RuntimeError: internal").to_payload()
    assert set(payload) == {"code", "message", "detail"}
    assert "Traceback" not in str(payload)


def test_the_payload_omits_an_absent_detail() -> None:
    assert set(ColdStartError("no history").to_payload()) == {"code", "message"}


def test_the_degradable_types_are_exactly_the_documented_ones() -> None:
    """Which failures serve the popularity fallback (FR-05, NFR-05)."""
    assert set(DEGRADATION_REASONS) == {
        ColdStartError,
        FeatureStoreUnavailableError,
        ModelUnavailableError,
    }


def test_a_user_not_found_is_not_degradable() -> None:
    """A 404 is an answer, not a degraded recommendation."""
    assert UserNotFoundError("nope").degradable is False


def test_pydantic_validation_error_is_the_sixth_row() -> None:
    """Table 10 row 1 is Pydantic's own type, deliberately not redefined."""
    from pydantic import ValidationError

    import errors

    assert not hasattr(errors, "ValidationError")
    assert issubclass(ValidationError, Exception)
