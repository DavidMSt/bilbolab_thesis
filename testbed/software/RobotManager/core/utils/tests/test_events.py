import threading
import time
from dataclasses import dataclass

import pytest

# Import everything from your NEW module here
# from your_module_path.events_new import ( ... )
from core.utils.events import (
    Event,
    EventFlag,
    Subscriber,
    SubscriberType,
    SubscriberListener,
    PatternSubscriber,
    AND,
    OR,
    wait_for_events,
    TIMEOUT,
    EventContainer,
    event_definition,
    pred_flag_equals,
    pred_flag_in,
    pred_flag_contains,
    pred_data_equals,
    pred_data_dict_key_equals,
    pred_data_in,
)
from core.utils.logging_utils import Logger


# ---------------------------
# Utility helpers for tests
# ---------------------------

def wait_true(pred, timeout=1.5, period=0.01):
    """Spin until pred() returns True or timeout elapses."""
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(period)
    return False


# ================================================================
# Basic Event.set + .wait behavior (stale window, typing, flags)
# ================================================================

def test_event_wait_with_flag_predicate_success():
    e = Event(id="e1", flags=[EventFlag(id="level", data_type=str)])

    def setter():
        time.sleep(0.05)
        e.set(data={"v": 1}, flags={"level": "high"})

    threading.Thread(target=setter, daemon=True).start()
    data, result = e.wait(predicate=pred_flag_equals("level", "high"), timeout=1.0)
    assert data == {"v": 1}
    assert result is not None
    assert result.match == e or result.match_id == e.uid


def test_event_wait_with_data_predicate_success():
    e = Event(id="e2")

    def setter():
        time.sleep(0.05)
        e.set(data={"state": "ready"})

    threading.Thread(target=setter, daemon=True).start()
    data, result = e.wait(predicate=pred_data_dict_key_equals("state", "ready"), timeout=1.0)
    assert data == {"state": "ready"}
    assert result is not None


def test_event_wait_stale_window_immediate():
    e = Event(id="e3", flags=[EventFlag(id="mode", data_type=str)])
    e.set(data={"x": 1}, flags={"mode": "auto"})
    time.sleep(0.1)  # still inside the stale window below
    data, result = e.wait(predicate=pred_flag_equals("mode", "auto"),
                          stale_event_time=1.0, timeout=0.1)
    assert data == {"x": 1}
    assert result is not None


def test_event_wait_timeout_returns_TIMEOUT():
    e = Event(id="e4", flags=[EventFlag(id="mode", data_type=str)])
    # emit "manual", but wait for "auto" with very short stale
    e.set(data={"x": 1}, flags={"mode": "manual"})
    time.sleep(0.1)
    data, result = e.wait(predicate=pred_flag_equals("mode", "auto"),
                          stale_event_time=0.01, timeout=0.1)
    assert data is TIMEOUT
    assert result is None


def test_invalid_flag_type_raises():
    e = Event(id="e5", flags=[EventFlag(id="level", data_type=str)])
    with pytest.raises(TypeError):
        e.set(flags={"level": 123})  # wrong type


def test_event_id_validation():
    # valid id
    Event(id="hello_world")
    # invalid: spaces or hyphens should raise
    with pytest.raises(ValueError):
        Event(id="bad id")
    with pytest.raises(ValueError):
        Event(id="bad-id")
    # invalid: empty
    with pytest.raises(ValueError):
        Event(id="")


# ================================================================
# History / stale window helpers mirror old semantics
# ================================================================

def test_has_match_in_window_uses_history_snapshot():
    e = Event(id="e6", flags=[EventFlag(id="lvl", data_type=str)])
    e.set(data={"n": 1}, flags={"lvl": "x"})
    time.sleep(0.02)
    e.set(data={"n": 2}, flags={"lvl": "z"})  # overwrite with non-matching

    pred = lambda f, d: d.get("n") == 1 and f.get("lvl") == "x"
    assert e.has_match_in_window(pred, window=0.5) is True


def test_first_match_in_window_returns_flags_and_data():
    e = Event(id="e7", flags=[EventFlag(id="tag", data_type=str)])
    payload = {"x": 42}
    e.set(data=payload, flags={"tag": "past"})
    time.sleep(0.05)
    out = e.first_match_in_window(pred_flag_equals("tag", "past"), window=0.5)
    assert out is not None
    flags, data = out
    assert flags == {"tag": "past"}
    assert data == payload


def test_history_pruning_respects_max_history_time():
    e = Event(id="e8")
    e.max_history_time = 0.02
    e.set(data=1)
    # ensure at least one history entry exists
    assert len(e.history) >= 1
    # simulate time passing
    future = time.monotonic() + 10.0
    e._prune_history(now=future)
    assert len(e.history) == 0


# ================================================================
# Copy semantics (copy on set vs aliasing) with new get_data()
# ================================================================

def test_copy_data_on_set_true_isolation_from_mutation():
    e = Event(id="e9", copy_data_on_set=True)
    src = {"k": ["a", "b"]}
    e.set(data=src)
    src["k"].append("c")
    stored = e.get_data(copy=False)
    assert stored == {"k": ["a", "b"]}


def test_copy_data_on_set_false_allows_aliasing():
    e = Event(id="e10", copy_data_on_set=False)
    src = {"k": ["a"]}
    e.set(data=src)
    src["k"].append("b")
    stored = e.get_data(copy=False)
    assert stored == {"k": ["a", "b"]}


# ================================================================
# Subscriber basics (single event), once semantics & callback shape
# ================================================================

def test_subscriber_once_and_callback_receives_match():
    e = Event(id="e11", flags=[EventFlag("mode", str)])
    hits = {"count": 0, "data": None}

    def cb(*, data, match):
        hits["count"] += 1
        hits["data"] = data
        # verify callback signature gives SubscriberMatch
        assert hasattr(match, "match_id")

    listener = e.on(callback=cb, predicate=pred_flag_equals("mode", "auto"), once=True)

    time.sleep(0.05)
    e.set(data={"v": "A"}, flags={"mode": "auto"})

    ok = wait_true(lambda: hits["count"] == 1, timeout=1.0)
    assert ok, "Listener should fire exactly once"
    assert hits["data"] == {"v": "A"}

    # Further events should not fire (once=True auto-stops)
    e.set(data={"v": "B"}, flags={"mode": "auto"})
    time.sleep(0.1)
    assert hits["count"] == 1
    listener.stop()


def test_subscriber_wait_mechanics_and_timeout():
    e = Event(id="e12")
    sub = Subscriber(events=[e], timeout=0.1)
    data, result = sub.wait()  # nothing emitted -> timeout
    assert data is TIMEOUT and result is None

    def emit():
        time.sleep(0.05)
        e.set(data=123)

    threading.Thread(target=emit, daemon=True).start()
    data, result = sub.wait(timeout=0.2)
    assert data == 123 and result is not None
    sub.stop()


# ================================================================
# Subscriber AND/OR semantics with multiple events
# ================================================================

def test_subscriber_and_waits_for_all_positions():
    e1 = Event(id="e13", flags=[EventFlag("a", str)])
    e2 = Event(id="e14")
    sub = Subscriber(events=[(e1, pred_flag_equals("a", "x")), e2],
                     type=SubscriberType.AND,
                     timeout=1.0)

    def emit():
        time.sleep(0.05)
        e1.set(data={"p": 1}, flags={"a": "x"})
        time.sleep(0.05)
        e2.set(data={"p": 2})

    threading.Thread(target=emit, daemon=True).start()
    data, match = sub.wait()
    assert match is not None
    assert isinstance(data, dict)
    assert data[e1] == {"p": 1}
    assert data[e2] == {"p": 2}
    sub.stop()


def test_subscriber_or_triggers_on_first_matching_child():
    e1 = Event(id="e15", flags=[EventFlag("k", str)])
    e2 = Event(id="e16")

    sub = Subscriber(events=[(e1, pred_flag_equals("k", "alpha")), e2],
                     type=SubscriberType.OR,
                     timeout=1.0)

    def emit():
        time.sleep(0.02)
        e2.set(data={"v": "B"})

    threading.Thread(target=emit, daemon=True).start()
    data, match = sub.wait()
    assert data == {"v": "B"}
    assert match is not None
    sub.stop()


# ================================================================
# PatternSubscriber: glob-attach and optional predicate
# ================================================================

def test_pattern_subscriber_attaches_and_receives_events():
    e1 = Event(id="sensor_temp")
    e2 = Event(id="sensor_humid")
    e3 = Event(id="actor_motor")

    hits = []

    ps = PatternSubscriber(pattern="sensor_*", predicate=None, stale_event_time=0.5)
    ps_listener = ps.on(lambda *, data, match: hits.append((match.match_id, data)), once=False)

    # Emit from both sensors, but not the actor
    e1.set(data={"t": 21})
    e2.set(data={"h": 55})
    e3.set(data={"rpm": 1000})

    ok = wait_true(lambda: len(hits) >= 2, timeout=1.0)
    assert ok
    # Ensure only sensor_* matched
    ids = [mid for (mid, _) in hits]
    assert "sensor_temp" in ids and "sensor_humid" in ids
    assert "actor_motor" not in ids

    ps_listener.stop()
    ps.stop()


def test_pattern_subscriber_predicate_filters():
    e1 = Event(id="log_info", flags=[EventFlag("level", str)])
    e2 = Event(id="log_warn", flags=[EventFlag("level", str)])

    hits = []

    # Only accept flags level in {"warn", "error"} for all matched events
    ps = PatternSubscriber(pattern="log_*",
                           predicate=pred_flag_in("level", {"warn", "error"}))
    ps_listener = ps.on(lambda *, data, match: hits.append((match.match_id, data)), once=False)

    e1.set(data="ok", flags={"level": "info"})  # should be ignored
    e2.set(data="be careful", flags={"level": "warn"})  # should pass

    ok = wait_true(lambda: len(hits) >= 1, timeout=1.0)
    assert ok
    assert hits[0][0] == "log_warn"
    assert hits[0][1] == "be careful"

    ps_listener.stop()
    ps.stop()


# ================================================================
# AND/OR factory helpers + wait_for_events
# ================================================================

def test_wait_for_events_or_with_pattern_and_predicate():
    a = Event(id="event1", flags=[EventFlag("level1", str)])
    b = Event(id="event2", flags=[EventFlag("level2", str)])
    c = Event(id="event3", flags=[EventFlag("level3", str)])

    def emit():
        time.sleep(0.05)
        a.set(data="A", flags={"level1": "nope"})
        b.set(data="B", flags={"level2": "ok"})  # <- this will win
        c.set(data="C", flags={"level3": "ok"})

    threading.Thread(target=emit, daemon=True).start()

    data, trace = wait_for_events(
        OR((a, pred_flag_equals("level1", "ok")), "event2", (c, pred_flag_equals("level3", "nope"))),
        timeout=1.0
    )
    assert trace is not None
    assert data == "B"


def test_wait_for_events_and_with_multiple_children():
    x = Event(id="x", flags=[EventFlag("t", str)])
    y = Event(id="y")

    def emit():
        time.sleep(0.02)
        x.set(data=1, flags={"t": "a"})
        time.sleep(0.02)
        y.set(data=2)

    threading.Thread(target=emit, daemon=True).start()

    data, trace = wait_for_events(AND((x, pred_flag_equals("t", "a")), y), timeout=1.0)
    assert trace is not None
    assert isinstance(data, dict)
    assert data[x] == 1 and data[y] == 2


# ================================================================
# Stale-window replay from saved matches (Subscriber.matches)
# ================================================================

def test_subscriber_stale_window_replays_recent_match():
    e = Event(id="e17", flags=[EventFlag("m", str)])
    sub = Subscriber(events=[(e, pred_flag_equals("m", "hit"))], save_matches=True, match_save_time=2.0)
    # Produce a match, then immediately wait with stale window
    e.set(data={"k": "v"}, flags={"m": "hit"})
    # No blocking: stale replay should return immediately
    data, trace = sub.wait(stale_event_time=1.0, timeout=0.01)
    assert data == {"k": "v"}
    assert trace is not None
    sub.stop()


# ================================================================
# SubscriberListener rate limiting (best-effort, non-flaky)
# ================================================================

def test_subscriber_listener_rate_limit_drops_bursts():
    e = Event(id="e18")
    calls = {"count": 0}

    def cb(*, data, match):
        calls["count"] += 1

    # 10 Hz max, we will emit ~30 events within ~0.2s -> expect small number of callbacks
    listener = e.on(callback=cb, once=False, max_rate=10.0)
    start = time.monotonic()
    for _ in range(30):
        e.set(data=time.monotonic())
        time.sleep(0.006)  # ~166 Hz burst

    # give a moment for deliveries
    time.sleep(0.3)
    listener.stop()

    # We expect far fewer than 30 callbacks with a 10Hz cap over ~0.2–0.3s window
    assert 1 <= calls["count"] < 10


# ================================================================
# Predicates: data vs flags helpers
# ================================================================

def test_pred_flag_contains_single_and_list():
    e = Event(id="e19", flags=[EventFlag("roles", (str, list, tuple, set))])

    e.set(data={"x": 1}, flags={"roles": "admin"})
    assert pred_flag_contains("roles", "admin")({"roles": "admin"}, {"x": 1}) is True

    e.set(data={"x": 2}, flags={"roles": ["user", "staff"]})
    assert pred_flag_contains("roles", "staff")({"roles": ["user", "staff"]}, {"x": 2}) is True
    assert pred_flag_contains("roles", "admin")({"roles": ["user", "staff"]}, {"x": 2}) is False


def test_pred_data_in_requires_dict():
    # pred_data_in should return False when data is not a dict
    assert pred_data_in("k", {"a", "b"})({}, "not-a-dict") is False
    # True when dict and value inside set
    assert pred_data_in("k", {"a", "b"})({}, {"k": "a"}) is True


# ================================================================
# Event containers + event_definition decorator behavior
# ================================================================

def test_event_definition_decorator_and_uids():
    @event_definition
    class MyEvents(EventContainer):
        ready: Event = Event(flags=[EventFlag("level", str)])
        moved: Event

    a = MyEvents(id="robotA")
    b = MyEvents(id="robotB")

    # fresh instances per object
    assert a.ready is not b.ready
    assert a.moved is not b.moved

    # children registered & parented
    assert "ready" in a.events and "moved" in a.events
    assert a.ready.parent is a and a.moved.parent is a

    # uid should include container id when present
    assert a.ready.uid == "robotA:ready"
    assert b.moved.uid == "robotB:moved"

    # flag schema preserved
    a.ready.set(flags={"level": "high"})
    # should not raise


def test_event_container_add_event_enforces_uniqueness_and_parent():
    c = EventContainer(id="C1")
    e1 = Event(id="x")
    c.add_event(e1)
    with pytest.raises(ValueError):
        c.add_event(e1)  # duplicate id in same container
    c2 = EventContainer(id="C2")
    with pytest.raises(ValueError):
        c2.add_event(e1)  # already has a parent


# ================================================================
# Integration: back-to-back sets deliver snapshot that matched
# ================================================================

def test_listener_receives_snapshot_that_matched_not_overwrite():
    e = Event(id="e20", flags=[EventFlag("level", str)])
    seen = []
    hit = threading.Event()

    def cb(*, data, match):
        seen.append(data)
        hit.set()

    listener = e.on(callback=cb, predicate=pred_flag_equals("level", "v1"), once=True)

    def producer():
        e.set(data={"k": "v1"}, flags={"level": "v1"})
        e.set(data={"k": "v2"}, flags={"level": "v2"})

    threading.Thread(target=producer, daemon=True).start()
    assert hit.wait(1.0) is True
    assert seen == [{"k": "v1"}]
    listener.stop()


if __name__ == '__main__':
    logger = Logger("TEST EVENTS")
    logger.info("-----")
    logger.info("Wait with flag predicate success")
    test_event_wait_with_flag_predicate_success()

    logger.info("-----")
    logger.info("Wait with data predicate success")
    test_event_wait_with_data_predicate_success()

    logger.info("-----")
    test_event_wait_stale_window_immediate()

    logger.info("-----")
    test_event_wait_timeout_returns_TIMEOUT()

    logger.info("-----")
    test_invalid_flag_type_raises()
    logger.info("-----")
    test_event_id_validation()
    logger.info("-----")
    test_has_match_in_window_uses_history_snapshot()
    logger.info("-----")
    test_first_match_in_window_returns_flags_and_data()
    logger.info("-----")
    test_history_pruning_respects_max_history_time()
    logger.info("-----")
    test_copy_data_on_set_true_isolation_from_mutation()
    logger.info("-----")
    test_copy_data_on_set_false_allows_aliasing()

    logger.info("-----")
    test_subscriber_once_and_callback_receives_match()

    logger.info("-----")
    test_subscriber_wait_mechanics_and_timeout()

    logger.info("-----")
    test_subscriber_and_waits_for_all_positions()
    logger.info("-----")
    test_subscriber_or_triggers_on_first_matching_child()

    logger.info("-----")
    test_pattern_subscriber_attaches_and_receives_events()
    logger.info("-----")
    test_pattern_subscriber_predicate_filters()

    logger.info("-----")
    test_wait_for_events_or_with_pattern_and_predicate()
    logger.info("-----")
    test_wait_for_events_and_with_multiple_children()

    logger.info("-----")
    test_subscriber_stale_window_replays_recent_match()
    logger.info("-----")
    test_subscriber_listener_rate_limit_drops_bursts()

    logger.info("-----")
    test_pred_flag_contains_single_and_list()
    logger.info("-----")
    test_pred_data_in_requires_dict()

    logger.info("-----")
    test_event_definition_decorator_and_uids()
    logger.info("-----")
    test_event_container_add_event_enforces_uniqueness_and_parent()
    logger.info("-----")
    test_listener_receives_snapshot_that_matched_not_overwrite()
    logger.info("-----")
    logger.info("All tests passed!")
