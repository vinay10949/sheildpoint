"""
In-memory stores for the claim intake pipeline (SP-203).

Two stores are maintained:

1. ``_CLAIMS_STORE`` — every claim that has been *accepted* (passed
   validation), keyed by claim ID. The IntakeAgent downstream reads from
   this.

2. ``_REVIEW_QUEUE`` — every claim that failed validation, in arrival
   order. Reviewers pull from the head of the list (FIFO).

Both stores are module-level dicts/lists so they survive across API
requests within a single process. In production these would be Postgres
tables + a Redis list, but the in-memory implementation is sufficient for
the MVP and the load-test AC.

Test isolation is provided by :func:`reset_stores` — tests should call it
in a fixture's teardown step (see ``conftest.py``).
"""

from __future__ import annotations

import threading
from typing import Any

from .schemas import (
    ClaimStatus,
    IntakeResult,
    IntakeSource,
    ReviewItem,
    StandardClaim,
)

# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
_CLAIMS_STORE: dict[str, dict[str, Any]] = {}
_REVIEW_QUEUE: list[ReviewItem] = []
_REVIEW_INDEX: dict[str, int] = {}  # claim_id → position in _REVIEW_QUEUE

#: Lock guarding both stores. The load test exercises the API concurrently,
#: so we need a real lock (not just the GIL — list.append is atomic but
#: the read-modify-write on _REVIEW_INDEX is not).
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Accepted claims
# ---------------------------------------------------------------------------
def put_accepted(claim: StandardClaim, claim_id: str, *, source: IntakeSource,
                 request_id: str | None = None, latency_sec: float = 0.0) -> IntakeResult:
    """Persist an accepted claim and return the success envelope."""
    with _LOCK:
        _CLAIMS_STORE[claim_id] = {
            "claim_id": claim_id,
            "status": ClaimStatus.ACCEPTED,
            "source": source,
            "claim": claim.model_dump(),
            "request_id": request_id,
            "latency_sec": latency_sec,
        }
    return IntakeResult(
        claim_id=claim_id,
        status=ClaimStatus.ACCEPTED,
        source=source,
        accepted=True,
        claim=claim,
        latency_sec=latency_sec,
        request_id=request_id,
    )


def get_accepted(claim_id: str) -> dict[str, Any] | None:
    """Return an accepted claim record, or None if not found."""
    with _LOCK:
        rec = _CLAIMS_STORE.get(claim_id)
        return dict(rec) if rec else None


def list_accepted(limit: int = 100) -> list[dict[str, Any]]:
    """List the most recent N accepted claims (insertion order)."""
    with _LOCK:
        items = list(_CLAIMS_STORE.values())
    return items[-limit:]


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
def put_review(item: ReviewItem) -> int:
    """Append a claim to the review queue. Returns its 1-indexed position."""
    with _LOCK:
        _REVIEW_QUEUE.append(item)
        pos = len(_REVIEW_QUEUE)
        _REVIEW_INDEX[item.claim_id] = pos - 1
    return pos


def get_review(claim_id: str) -> ReviewItem | None:
    """Look up a review item by claim ID."""
    with _LOCK:
        idx = _REVIEW_INDEX.get(claim_id)
        if idx is None:
            return None
        return _REVIEW_QUEUE[idx]


def list_review(limit: int = 100, status: ClaimStatus | None = None) -> list[ReviewItem]:
    """List review items, newest-first by default.

    If ``status`` is given, filter to that status only.
    """
    with _LOCK:
        items = list(_REVIEW_QUEUE)
    if status is not None:
        items = [it for it in items if it.status == status]
    return list(reversed(items))[:limit]


def next_review() -> ReviewItem | None:
    """Pop and return the oldest review item (FIFO), or None if empty.

    Marks the item as consumed by removing it from the queue. The reviewer
    is expected to take action and then either accept the claim (calling
    :func:`put_accepted`) or reject it (calling :func:`reject_review`).
    """
    with _LOCK:
        if not _REVIEW_QUEUE:
            return None
        item = _REVIEW_QUEUE.pop(0)
        # Rebuild index after pop
        _REVIEW_INDEX.clear()
        for i, it in enumerate(_REVIEW_QUEUE):
            _REVIEW_INDEX[it.claim_id] = i
        return item


def resolve_review(claim_id: str, *, accepted_claim: StandardClaim | None,
                   source: IntakeSource, request_id: str | None = None,
                   latency_sec: float = 0.0) -> IntakeResult:
    """Resolve a review item: either promote to accepted, or mark rejected.

    If ``accepted_claim`` is None, the item is marked rejected. The item is
    removed from the queue either way (reviewers don't re-pick resolved items).
    """
    with _LOCK:
        item = _REVIEW_INDEX.pop(claim_id, None)
        if item is None:
            # Not in the queue — nothing to do.
            pass
        else:
            # Remove from queue (linear scan, but queue is small in MVP)
            _REVIEW_QUEUE.pop(item)
            # Rebuild index
            _REVIEW_INDEX.clear()
            for i, it in enumerate(_REVIEW_QUEUE):
                _REVIEW_INDEX[it.claim_id] = i

    if accepted_claim is not None:
        return put_accepted(
            accepted_claim, claim_id, source=source,
            request_id=request_id, latency_sec=latency_sec,
        )
    # Rejected
    return IntakeResult(
        claim_id=claim_id,
        status=ClaimStatus.REJECTED,
        source=source,
        accepted=False,
        request_id=request_id,
        latency_sec=latency_sec,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def reset_stores() -> None:
    """Clear both stores. Call this in test fixtures for isolation."""
    with _LOCK:
        _CLAIMS_STORE.clear()
        _REVIEW_QUEUE.clear()
        _REVIEW_INDEX.clear()


def store_stats() -> dict[str, int]:
    """Return a small stats dict — useful for the /health endpoint."""
    with _LOCK:
        return {
            "accepted_count": len(_CLAIMS_STORE),
            "review_queue_depth": len(_REVIEW_QUEUE),
        }
