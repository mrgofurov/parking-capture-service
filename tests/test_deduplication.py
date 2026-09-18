import time
import pytest
from app.parking.deduplication import DeduplicationManager


def test_deduplication_cooldown():
    dedup = DeduplicationManager(cooldown_seconds=1)

    # First event should not be duplicate
    assert dedup.is_duplicate("entry-01", "entry", "01A777AA") is False

    # Record event
    dedup.record_event("entry-01", "entry", "01A777AA")

    # Immediate second check should be duplicate
    assert dedup.is_duplicate("entry-01", "entry", "01A777AA") is True

    # Different direction or different camera should not be duplicate
    assert dedup.is_duplicate("entry-01", "exit", "01A777AA") is False
    assert dedup.is_duplicate("entry-02", "entry", "01A777AA") is False

    # After cooldown expires, should no longer be duplicate
    time.sleep(1.1)
    assert dedup.is_duplicate("entry-01", "entry", "01A777AA") is False
