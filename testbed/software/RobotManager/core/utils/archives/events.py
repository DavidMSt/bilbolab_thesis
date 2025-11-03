from __future__ import annotations

import queue
import threading
from copy import deepcopy
import time
from dataclasses import is_dataclass, dataclass
from typing import Callable, Any, Optional, Union
from collections import deque
import weakref

# === CUSTOM MODULES ===================================================================================================
from core.utils.callbacks import callback_definition, CallbackContainer, Callback
from core.utils.dataclass_utils import deepcopy_dataclass
from core.utils.dict_utils import optimized_deepcopy
from core.utils.logging_utils import Logger
from core.utils.singleton import _SingletonMeta

# === GLOBAL VARIABLES =================================================================================================
logger = Logger('events')


# === EVENT FLAG =======================================================================================================
class EventFlag:
    id: str
    types: tuple[type, ...]  # always a tuple at runtime

    # === INIT =========================================================================================================
    def __init__(self, id: str, data_type: type | tuple[type, ...]):
        self.id = id
        if isinstance(data_type, tuple):
            if not data_type or not all(isinstance(t, type) for t in data_type):
                raise TypeError("data_type tuple must be non-empty and contain only types.")
            self.types = data_type
        elif isinstance(data_type, type):
            self.types = (data_type,)
        else:
            raise TypeError("data_type must be a type or tuple[type, ...]")

    # ------------------------------------------------------------------------------------------------------------------
    def accepts(self, value: Any) -> bool:
        return isinstance(value, self.types)

    # ------------------------------------------------------------------------------------------------------------------
    def describe(self) -> str:
        return " | ".join(t.__name__ for t in self.types)


# === PREDICATE ========================================================================================================
Predicate = Callable[[dict[str, Any], Any], bool]  # (flags, data) -> bool


def pred_flag_in(key, values) -> Predicate:
    return lambda f, d: f.get(key) in values


def pred_data_in(key, values) -> Predicate:
    return lambda f, d: d.get(key) in values


def pred_data_dict_key_equals(key, expected) -> Predicate:
    return lambda f, d: d.get(key) == expected


def pred_data_equals(expected) -> Predicate:
    return lambda f, d: d == expected


def pred_flag_equals(key, expected) -> Predicate:
    return lambda f, d: f.get(key) == expected


def pred_flag_contains(flag_key: str, match_value: Any) -> Predicate:
    """
    Returns a predicate that checks if `match_value` is present in the
    flag list (or equals the single flag value) for `flag_key`.

    Works whether the flag value is a list/tuple/set or a single value.
    """

    def _pred(flags, data):
        if flag_key not in flags:
            return False
        val = flags[flag_key]
        if isinstance(val, (list, tuple, set)):
            return match_value in val
        return val == match_value

    return _pred


# === EVENT ============================================================================================================
@callback_definition
class EventCallbacks:
    set: CallbackContainer


class Event:
    flags: dict[str, EventFlag]

    data: Any
    callbacks: EventCallbacks
    data_type: type | None
    history: deque[tuple[float, dict[str, Any], Any]]

    copy_data_on_set = True

    copy_fn = None

    max_history_time: float = 2.0

    dict_copy_cache = None

    # === INIT =========================================================================================================
    def __init__(self,
                 data_type: type | None = None,
                 flags: EventFlag | list[EventFlag] = None,
                 copy_data_on_set: bool = True,
                 data_is_static_dict: bool = False, ):

        if flags is None:
            flags = []

        if not isinstance(flags, list):
            flags = [flags]

        self.flags = {}

        for flag in flags:
            self.flags[flag.id] = flag

        self.callbacks = EventCallbacks()
        self.data = None

        self.data_type = data_type
        self.copy_data_on_set = copy_data_on_set
        self.data_is_static_dict = data_is_static_dict

        self.history: deque[tuple[float, dict[str, Any], Any]] = deque()
        self._history_lock = threading.Lock()

        active_event_loop.addEvent(self)

    # === METHODS ======================================================================================================
    def on(self,
           callback: Callback | Callable,
           predicate: Predicate = None,
           once: bool = False,
           stale_event_time=None,
           timeout=None,
           input_data=True,
           max_rate=None) -> EventListener | OneShotHandle:

        # One-shot special case: keep using a thread that waits and fires once.
        if once and stale_event_time is not None:
            return self._spawnOneShotListener(callback, predicate, stale_event_time, input_data, timeout)

        # Continuous (or one-shot without stale) listener backed by queue waiter
        listener = EventListener(event=self,
                                 predicate=predicate,
                                 callback=callback,
                                 once=once,
                                 spawn_thread=False,
                                 max_rate=max_rate,
                                 input_data=input_data,
                                 stale_event_time=stale_event_time)
        listener.start()
        return listener

    # ------------------------------------------------------------------------------------------------------------------
    def wait(self, predicate: Predicate = None, timeout: float = None, stale_event_time: float = None) -> bool:
        waiter = EventWaiter(events=self,
                             predicates=predicate,
                             timeout=timeout,
                             stale_event_time=stale_event_time,
                             wait_for_all=False, )
        return waiter.wait()

    # ------------------------------------------------------------------------------------------------------------------
    def set(self, data=None, flags: dict = None) -> None:

        # Check if the flags are valid
        assert (isinstance(flags, dict) or flags is None)
        flags = flags or {}

        # Check if all flags are valid
        for flag in flags:
            if flag not in self.flags:
                raise ValueError(f"Invalid flag: {flag}")

            ef = self.flags[flag]
            value = flags[flag]

            if not ef.accepts(value):
                raise TypeError(
                    f"Flag '{flag}' is expected to be of type {ef.describe()}, "
                    f"but got {type(value).__name__} instead."
                )

        # Check if the data is valid
        if data is not None:
            if self.data_type is not None:
                if not isinstance(data, self.data_type):
                    raise TypeError(
                        f"Data is expected to be of type {self.data_type.__name__}, "
                        f"but got {type(data).__name__} instead."
                    )

        # Make a copy of the data
        if self.copy_data_on_set:
            payload = self._copy_payload(data)
        else:
            payload = data

        self.data = payload

        now = time.monotonic()
        flags = dict(flags)
        with self._history_lock:
            self.history.append((now, flags, payload))
            self._prune_history(now)

        self.callbacks.set.call(data=payload, flags=flags)

    # ------------------------------------------------------------------------------------------------------------------
    def getData(self, copy: bool = True) -> Any:
        if copy:
            return deepcopy(self.data)
        else:
            return self.data

    # ------------------------------------------------------------------------------------------------------------------
    def has_match_in_window(self, predicate: Predicate | None, window: float, now: float | None = None) -> bool:
        if window is None or window <= 0:
            return False
        if now is None:
            now = time.monotonic()
        cutoff = now - window
        with self._history_lock:
            self._prune_history(now)
            for ts, flags, data in reversed(self.history):
                if ts < cutoff:
                    break
                if predicate is None or predicate(flags, data):
                    return True
        return False

    # ------------------------------------------------------------------------------------------------------------------
    def first_match_in_window(self, predicate: Predicate | None, window: float, now: float | None = None):
        if window is None or window <= 0:
            return None
        if now is None:
            now = time.monotonic()
        cutoff = now - window
        with self._history_lock:
            self._prune_history(now)
            for ts, flags, data in reversed(self.history):
                if ts < cutoff:
                    break
                if predicate is None or predicate(flags, data):
                    return flags, data
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def _spawnOneShotListener(self,
                              callback: Callback | Callable,
                              predicate: Predicate,
                              stale_event_time: float,
                              input_data: bool,
                              timeout: float) -> OneShotHandle:

        waiter = EventWaiter(events=self, predicates=predicate, timeout=timeout, stale_event_time=stale_event_time)

        def listener_thread():
            if waiter.wait():
                if input_data:
                    if len(waiter.matched_payloads) == 1:
                        _, d = waiter.matched_payloads[0]
                        callback(d)
                    else:
                        assembled = {}
                        for i, ev in enumerate(waiter.events):
                            _, d = waiter.matched_payloads[i]
                            assembled[ev] = d
                        callback(assembled)
                else:
                    callback()

        t = threading.Thread(target=listener_thread, daemon=True)
        t.start()
        return OneShotHandle(waiter, t)

    # === PRIVATE METHODS ==============================================================================================
    def _prune_history(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.max_history_time
        dq = self.history
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    # ------------------------------------------------------------------------------------------------------------------`
    def _copy_payload(self, data) -> Any:
        # if data is None:
        #     return None
        # if self.copy_fn is not None:
        #     return self.copy_fn(data)
        # if self.copy_data_on_set:
        #     return deepcopy(data)
        # return data
        try:
            if self.data_is_static_dict and self.data_type is dict:
                payload, self.dict_copy_cache = optimized_deepcopy(data, self.dict_copy_cache)
            elif is_dataclass(self.data_type):
                payload = deepcopy_dataclass(data)
            else:
                payload = deepcopy(data)
        except Exception as e:
            payload = data
            self.copy_data_on_set = False
            logger.warning(f"Could not copy data for event {self}: {e}")

        return payload


# === EVENT WAITER =====================================================================================================
class EventWaiter:
    """
    Queue-backed waiter.
    - Maintains per-waiter queue of matched snapshots.
    - Supports once (auto-remove after first match) or continuous modes.
    - Preserves stale-event precheck and exact (flags, data) snapshots per position.
    """
    events: list[Event]
    predicates: list[Predicate] | None
    timeout: float | None
    stale_event_time: float | None
    wait_for_all: bool
    once: bool

    events_finished: list[bool]
    matched_payloads: list[tuple[dict[str, Any] | None, Any | None]]

    _abort: bool
    _queue: queue.Queue  # holds snapshots of matched_payloads

    data: Any = None  # convenience for single-event waits

    _SENTINEL = object()

    def __init__(self,
                 events: Event | list[Event],
                 predicates: Predicate | list[Predicate] | None = None,
                 timeout: float | None = None,
                 stale_event_time: float | None = None,
                 wait_for_all: bool = False,
                 once: bool = True,
                 queue_maxsize: int = 0):
        """
        queue_maxsize: 0 => unbounded; else bounded with drop-oldest policy on overflow.
        """
        if not isinstance(events, list):
            events = [events]
        if predicates is not None and not isinstance(predicates, list):
            predicates = [predicates]

        self.events = events
        self.predicates = predicates
        self.timeout = timeout
        self.stale_event_time = stale_event_time
        self.wait_for_all = wait_for_all
        self.once = once

        n = len(self.events)
        self.events_finished = [False] * n
        self.matched_payloads = [(None, None)] * n

        self._abort = False
        self._queue = queue.Queue(maxsize=queue_maxsize)

        # Register with the active loop immediately (not in wait())
        active_event_loop.addWaiter(self)

    # --- Public API ------------------------------------------------------------
    def wait(self, timeout: float | None = None) -> bool:
        """
        Block until a matched snapshot is available or timed out/aborted.
        Returns True if a snapshot was delivered into matched_payloads; False otherwise.
        """
        if timeout is None:
            timeout = self.timeout

        # If already aborted and nothing to consume, return immediately
        if self._abort and self._queue.empty():
            return False

        try:
            snap = self._queue.get(timeout=timeout) if timeout is not None else self._queue.get()
        except queue.Empty:
            return False

        # Wake-up sentinel injected by stop()
        if snap is self._SENTINEL:
            return False

        # Snap is a list of (flags, data) matching positions
        self.matched_payloads = snap

        # Convenience for single-event case
        if len(snap) == 1:
            _, d = snap[0]
            self.data = d
        else:
            self.data = None

        return True

    def stop(self):
        """Abort future waiting, deregister from loop, and wake any blocking wait()."""
        self._abort = True
        active_event_loop.removeWaiter(self)

        # Push sentinel to wake any blocking .wait()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()  # drop oldest to make room
            except queue.Empty:
                pass
            self._queue.put_nowait(self._SENTINEL)

    # --- Called by EventLoop under lock ---------------------------------------
    def _set_match(self, pos: int, flags: dict[str, Any], data: Any):
        self.matched_payloads[pos] = (flags, data)

    def _enqueue_snapshot_and_prepare_next(self):
        """Enqueue a snapshot of current matches; reset or finish based on 'once'."""
        # Snapshot current matched payloads
        snap = list(self.matched_payloads)

        # Non-blocking put with drop-oldest policy if bounded and full
        try:
            self._queue.put_nowait(snap)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait(snap)

        if self.once:
            # Mark for removal; EventLoop will remove us after returning from dispatch
            self._abort = True
            return "finish"
        else:
            # Reset for next round
            self.events_finished = [False] * len(self.events)
            self.matched_payloads = [(None, None)] * len(self.events)
            return "continue"


# === EVENT LISTENER ===================================================================================================
class OneShotHandle:
    def __init__(self, waiter: EventWaiter, thread: threading.Thread):
        self._waiter = waiter
        self._thread = thread

    def cancel(self):
        self._waiter.stop()

    def join(self, timeout=None):
        self._thread.join(timeout)


# === FUNCTIONS ========================================================================================================


# === EVENT LISTENER ===================================================================================================
class EventListener:
    """
    Continuous listener built on a single queue-backed EventWaiter.

    Adds:
      - max_rate (Hz) throttling: limits callback invocations to at most N times per second.
      - Burst coalescing: when throttled, only the most recent snapshot is kept (queue size = 1).

    Behavior:
      - If max_rate is None or <= 0, no throttling and the queue is unbounded (legacy behavior).
      - If max_rate > 0, the listener sleeps as needed between deliveries to enforce the rate.
        While sleeping, the waiter's queue (size 1) drops older snapshots in favor of the newest,
        so the next delivery after the sleep reflects the latest state.
    """
    events: list[Event]
    predicate: list[Predicate] | None
    callback: Callback | Callable
    once: bool
    wait_for_all: bool
    max_rate: float | None
    input_data: bool
    spawn_thread: bool
    stale_event_time: float | None

    _waiter: EventWaiter | None
    _exit: bool
    thread: threading.Thread

    # rate limiting fields
    _min_interval: float
    _last_emit: float

    def __init__(self, event: Event | list[Event],
                 predicate: Predicate | list[Predicate] | None = None,
                 callback: Callback | Callable | None = None,
                 once: bool = False,
                 wait_for_all: bool = False,
                 max_rate: float | None = None,
                 spawn_thread: bool = False,
                 input_data: bool = True,
                 stale_event_time: float | None = 0.05,
                 queue_maxsize: int | None = None):
        if not isinstance(event, list):
            event = [event]
        if predicate is not None and not isinstance(predicate, list):
            predicate = [predicate]

        self.events = event
        self.predicate = predicate
        self.callback = callback
        self.input_data = input_data
        self.max_rate = max_rate
        self.wait_for_all = wait_for_all
        self.once = once
        self.spawn_thread = spawn_thread
        self.stale_event_time = stale_event_time

        # --- Rate limiting setup ---
        self._min_interval = (1.0 / max_rate) if (max_rate and max_rate > 0) else 0.0
        self._last_emit = 0.0

        # If the caller explicitly passed queue_maxsize, respect it; otherwise:
        # - when throttling -> 1 (coalesce to latest)
        # - when not throttling -> 0 (unbounded; legacy behavior)
        if queue_maxsize is None:
            effective_qsize = 1 if self._min_interval > 0.0 else 0
        else:
            effective_qsize = queue_maxsize

        self._waiter = EventWaiter(
            events=self.events,
            predicates=self.predicate,
            timeout=None,
            stale_event_time=self.stale_event_time,
            wait_for_all=self.wait_for_all,
            once=self.once is True,  # one-shot listener → one-shot waiter
            queue_maxsize=effective_qsize,  # <-- important for coalescing
        )
        self._exit = False
        self.thread = threading.Thread(target=self._task, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._exit = True
        if self._waiter is not None:
            self._waiter.stop()
        if self.thread.is_alive():
            self.thread.join()

    # --- Private ---------------------------------------------------------------
    def _deliver_callback(self):
        """Assemble data from the waiter's matched payloads and invoke callback."""
        if len(self.events) == 1:
            _, data = self._waiter.matched_payloads[0]
            payload = data if self.input_data else None
        else:
            assembled = {}
            for i, ev in enumerate(self.events):
                _, d = self._waiter.matched_payloads[i]
                assembled[ev] = d
            payload = assembled if self.input_data else None

        if self.spawn_thread:
            th = threading.Thread(
                target=self.callback, args=(() if payload is None else (payload,)), daemon=True
            )
            th.start()
        else:
            self.callback() if payload is None else self.callback(payload)

    def _task(self):
        while not self._exit:
            ok = self._waiter.wait(timeout=None)
            if not ok:
                break

            # --- Rate limiting: enforce minimum spacing between callback invocations
            if self._min_interval > 0.0 and self._last_emit > 0.0:
                now = time.monotonic()
                remaining = self._min_interval - (now - self._last_emit)
                if remaining > 0:
                    # Sleep to enforce the interval. During this sleep, the waiter's queue
                    # (size 1) will keep only the latest snapshot if more events arrive.
                    time.sleep(remaining)

            self._deliver_callback()
            if self._min_interval > 0.0:
                self._last_emit = time.monotonic()

            if self.once:
                break


@dataclass(frozen=True)
class EventMatch:
    """Snapshot describing which event matched and with what payload."""
    event: Event
    index: int  # position in the input 'events' list
    flags: dict[str, Any]
    data: Any  # deep-copied snapshot captured by the waiter


@dataclass(frozen=True)
class WaitResult:
    """High-level result of waiting on one or more events."""
    ok: bool  # True if condition satisfied (any or all)
    timeout: bool  # True if we timed out
    finished: list[bool]  # per-position matched booleans (like before)
    matches: list[EventMatch]  # snapshots for all matched positions
    first: Optional[EventMatch]  # convenience: first matched position or None

    def matched(self, event: "Event") -> Optional[EventMatch]:
        """Find the match for a specific Event (first occurrence) if present."""
        for m in self.matches:
            if m.event is event:
                return m
        return None


def waitForEvents(
        events: Event | list[Event],
        predicates: Predicate | list[Predicate] = None,
        *,
        timeout: Optional[float] = None,
        wait_for_all: bool = False,
        stale_event_time: Optional[float] = None,
) -> WaitResult:
    """
    Like waitForEvents, but:
      - supports stale_event_time (catch events that fired just before waiting)
      - returns a WaitResult with exact matched snapshots.

    NOTE: Doesn't spawn threads; uses a one-shot EventWaiter internally.
    """
    if not isinstance(events, list):
        events = [events]
    if predicates is not None and not isinstance(predicates, list):
        predicates = [predicates]

    waiter = EventWaiter(
        events=events,
        predicates=predicates,  # may be None or fewer than events
        timeout=timeout,
        stale_event_time=stale_event_time,  # <-- new goodness
        wait_for_all=wait_for_all,
        once=True,
        queue_maxsize=0,
    )
    try:
        ok = waiter.wait(timeout=None)  # waiter has its own timeout
        finished = list(waiter.events_finished)
        timeout_hit = not ok

        matches: list[EventMatch] = []
        if ok:
            for idx, done in enumerate(finished):
                if done:
                    flags, data = waiter.matched_payloads[idx]
                    matches.append(EventMatch(
                        event=events[idx], index=idx, flags=flags or {}, data=data
                    ))

        return WaitResult(
            ok=ok,
            timeout=timeout_hit,
            finished=finished,
            matches=matches,
            first=(matches[0] if matches else None),
        )
    finally:
        waiter.stop()


# === EVENT LOOP =================================================================================================
class EventLoop(metaclass=_SingletonMeta):
    """
    Scalable event dispatcher with:
      - per-waiter queues (push on match)
      - stale-window precheck on registration
      - exact snapshot delivery
    """

    def __init__(self):
        self.events = weakref.WeakSet()
        self.waiters: list[EventWaiter] = []
        self.waiters_by_event: dict[Event, set[EventWaiter]] = {}
        self._waiters_lock = threading.RLock()

    # -- Event registration -----------------------------------------------------
    def addEvent(self, event: Event):
        with self._waiters_lock:
            if event in self.events:
                return
            self.events.add(event)
            event.callbacks.set.register(Callback(
                function=self._event_set,
                inputs={'event': event}
            ))

    # -- Waiter registration/removal -------------------------------------------
    def addWaiter(self, waiter: EventWaiter):
        to_finish = []
        with self._waiters_lock:
            self.waiters.append(waiter)
            for ev in waiter.events:
                self.waiters_by_event.setdefault(ev, set()).add(waiter)

            # Ensure history covers stale window
            if waiter.stale_event_time and waiter.stale_event_time > 0:
                for ev in waiter.events:
                    if getattr(ev, "max_history_time", 0) < waiter.stale_event_time:
                        ev.max_history_time = waiter.stale_event_time

                # Pre-check recent history and enqueue immediately if satisfied
                now = time.monotonic()
                for i, ev in enumerate(waiter.events):
                    pred = waiter.predicates[i] if (waiter.predicates and i < len(waiter.predicates)) else None
                    match = ev.first_match_in_window(pred, waiter.stale_event_time, now=now)
                    if match is not None:
                        flags, data = match
                        waiter.events_finished[i] = True
                        waiter._set_match(i, flags, data)

                satisfied = all(waiter.events_finished) if waiter.wait_for_all else any(waiter.events_finished)
                if satisfied:
                    state = waiter._enqueue_snapshot_and_prepare_next()
                    if state == "finish":
                        to_finish.append(waiter)

        # Clean-up for one-shot satisfied during precheck
        if to_finish:
            with self._waiters_lock:
                for w in to_finish:
                    self._unsafe_removeWaiter(w)

    def removeWaiter(self, waiter: EventWaiter):
        with self._waiters_lock:
            self._unsafe_removeWaiter(waiter)

    def _unsafe_removeWaiter(self, waiter: EventWaiter):
        if waiter in self.waiters:
            self.waiters.remove(waiter)
        for ev in getattr(waiter, "events", ()):
            s = self.waiters_by_event.get(ev)
            if s is not None:
                s.discard(waiter)
                if not s:
                    self.waiters_by_event.pop(ev, None)

    # -- Dispatch ---------------------------------------------------------------
    def _event_set(self, data, event: Event, flags, *args, **kwargs):
        to_finish = []
        with self._waiters_lock:
            watchers = list(self.waiters_by_event.get(event, ()))
            for waiter in watchers:
                if waiter._abort:
                    continue

                matched_any_pos = False
                for pos, ev in enumerate(waiter.events):
                    if ev is not event:
                        continue
                    pred = (waiter.predicates[pos]
                            if (waiter.predicates and pos < len(waiter.predicates))
                            else None)
                    ok = pred(flags, data) if pred is not None else True
                    if ok:
                        matched_any_pos = True
                        if not waiter.events_finished[pos]:
                            waiter.events_finished[pos] = True
                            waiter._set_match(pos, flags, data)

                if not matched_any_pos:
                    continue

                satisfied = all(waiter.events_finished) if waiter.wait_for_all else any(waiter.events_finished)
                if satisfied:
                    state = waiter._enqueue_snapshot_and_prepare_next()
                    if state == "finish":
                        to_finish.append(waiter)

            # Remove any finished one-shot waiters from indices
            if to_finish:
                for w in to_finish:
                    self._unsafe_removeWaiter(w)


def event_definition(cls):
    """
    Decorator to make Event fields independent per instance.

    - If an attribute is annotated as `Event` (or Optional[Event]/Union[..., Event, ...])
      and the instance doesn't set it in __init__, create a fresh Event for the instance.
    - If a class-level default value is an Event, clone it per instance, preserving flags.
    """
    import sys
    import typing
    from typing import get_origin, get_args

    original_init = getattr(cls, "__init__", None)

    # Resolve annotations, supporting `from __future__ import annotations`
    try:
        module_globals = sys.modules[cls.__module__].__dict__
        hints = typing.get_type_hints(cls, globalns=module_globals, localns=dict(vars(cls)))
    except Exception:
        hints = getattr(cls, "__annotations__", {}) or {}

    def _is_event_type(t) -> bool:
        if t is Event:
            return True
        if isinstance(t, str):
            return t == "Event" or t.endswith(".Event")
        origin = get_origin(t)
        # typing.Union (Py3.8/3.9) and PEP 604 unions (a | b) use different origins
        try:
            import types as _types
            is_union = (origin is typing.Union) or (origin is _types.UnionType)
        except Exception:
            is_union = (origin is typing.Union)
        if is_union:
            return any(_is_event_type(arg) for arg in get_args(t))
        if origin is typing.ClassVar:
            return False
        return False

    def _clone_event_template(template: Event) -> Event:
        # Preserve declared flag schema
        flags = [EventFlag(ef.id, ef.types) for ef in template.flags.values()]
        clone = Event(
            data_type=template.data_type,
            flags=flags,
            copy_data_on_set=template.copy_data_on_set,
        )
        # Copy non-ctor attrs too
        clone.copy_fn = template.copy_fn
        clone.max_history_time = template.max_history_time
        return clone

    def new_init(self, *args, **kwargs):
        if original_init:
            original_init(self, *args, **kwargs)

        # Annotated attributes
        if isinstance(hints, dict):
            for attr_name, anno in hints.items():
                default_val = getattr(cls, attr_name, None)
                if isinstance(default_val, Event):
                    # Replace class-level Event with a per-instance clone
                    setattr(self, attr_name, _clone_event_template(default_val))
                    continue
                if _is_event_type(anno) and attr_name not in self.__dict__:
                    # No default provided; create a fresh Event
                    setattr(self, attr_name, Event())

        # Unannotated class-level Event defaults
        for attr_name, value in vars(cls).items():
            if isinstance(value, Event) and attr_name not in self.__dict__:
                setattr(self, attr_name, _clone_event_template(value))

    cls.__init__ = new_init
    return cls


active_event_loop = EventLoop()


# === EXAMPLES =========================================================================================================
def test_wait_for_flag_predicate():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    fired = []

    def setter():
        time.sleep(0.1)
        e.set(data={"v": 1}, flags={"level": "high"})

    threading.Thread(target=setter, daemon=True).start()
    ok = e.wait(predicate=pred_flag_equals("level", "high"), timeout=1.0)
    assert ok is True, "Event.wait with flag predicate should succeed"
    assert e.getData() == {"v": 1}, "Event data should be set"


def test_wait_for_data_predicate():
    e = Event()

    def setter():
        time.sleep(0.1)
        e.set(data={"state": "ready"})

    threading.Thread(target=setter, daemon=True).start()
    ok = e.wait(predicate=pred_data_dict_key_equals("state", "ready"), timeout=1.0)
    assert ok is True, "Event.wait with data predicate should succeed"


def test_stale_event_wait_success():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    e.set(data={"x": 1}, flags={"level": "high"})
    time.sleep(0.2)  # event happened in the recent past
    ok = e.wait(predicate=pred_flag_equals("level", "high"),
                stale_event_time=1.0, timeout=5)
    assert ok is True, "Stale-window wait should immediately succeed for recent event"


def test_stale_event_wait_timeout():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    e.set(data={"x": 1}, flags={"level": "high"})
    time.sleep(0.3)
    ok = e.wait(predicate=pred_flag_equals("level", "high"),
                stale_event_time=0.05, timeout=0.2)
    assert ok is False, "Stale-window wait should fail if event is outside window"


def test_multiple_events_wait_for_all():
    e1 = Event(flags=[EventFlag(id="level", data_type=str)])
    e2 = Event()

    def setter():
        time.sleep(0.05)
        e1.set(data=1, flags={"level": "high"})
        time.sleep(0.05)
        e2.set(data=2)

    threading.Thread(target=setter, daemon=True).start()

    w = EventWaiter(events=[e1, e2],
                    predicates=[pred_flag_equals("level", "high"), None],
                    timeout=1.0, wait_for_all=True)
    ok = w.wait()
    assert ok is True, "Waiter across two events (wait_for_all) should succeed"
    assert w.events_finished == [True, True]


def test_event_listener_basic_once():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    hit = threading.Event()
    seen = []

    def cb(data):
        seen.append(data)
        hit.set()

    listener = e.on(callback=cb,
                    predicate=pred_flag_equals("level", "high"),
                    once=True,
                    input_data=True)

    time.sleep(0.05)
    e.set(data=42, flags={"level": "high"})
    assert hit.wait(1.0) is True, "Listener should fire once"
    # assert seen == [42], "Listener callback should receive event data"


def test_one_shot_listener_with_stale():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    # Fire in the past
    e.set(data=99, flags={"level": "high"})
    time.sleep(0.1)

    hit = threading.Event()

    def cb(data):
        assert data == 99
        hit.set()

    # Should trigger immediately thanks to stale_event_time
    e.on(callback=cb, predicate=pred_flag_equals("level", "high"),
         once=True, stale_event_time=1.0, timeout=0.5)
    assert hit.wait(0.2) is True, "One-shot listener should trigger immediately from stale window"


def test_duplicate_event_entries_different_predicates_wait_for_all():
    # Same Event watched twice with different predicates
    e = Event(flags=[EventFlag(id="a", data_type=str),
                     EventFlag(id="b", data_type=str)])

    # waiter waits for both conditions on the SAME event
    w = EventWaiter(events=[e, e],
                    predicates=[pred_flag_equals("a", "x"),
                                pred_flag_equals("b", "y")],
                    timeout=1.0, wait_for_all=True)

    def setter():
        # First satisfies only position 0
        time.sleep(0.05)
        e.set(flags={"a": "x"})
        # Then satisfies only position 1
        time.sleep(0.05)
        e.set(flags={"b": "y"})

    threading.Thread(target=setter, daemon=True).start()
    ok = w.wait()
    assert ok is True, "Waiter should satisfy after both predicates matched on same Event"
    assert w.events_finished == [True, True]


def test_immediate_notify_stale_precheck_with_predicate():
    e = Event(flags=[EventFlag(id="mode", data_type=str)])
    e.set(flags={"mode": "auto"})
    time.sleep(0.05)  # still within stale window

    start = time.monotonic()
    w = EventWaiter(events=e,
                    predicates=pred_flag_equals("mode", "auto"),
                    stale_event_time=0.5, timeout=1.0)
    ok = w.wait()
    elapsed = time.monotonic() - start
    assert ok is True, "Immediate notify via stale precheck should succeed"
    assert elapsed < 0.1, "Wait should return almost immediately when satisfied by history"


def test_event_definition_decorator():
    @event_definition
    class MyEvents:
        event1: Event = Event(flags=EventFlag(id="level", data_type=str))
        event2: Event

    a = MyEvents()
    b = MyEvents()
    # Each instance gets distinct Event objects
    assert a.event1 is not b.event1, "event1 should be unique per instance"
    assert a.event2 is not b.event2, "event2 should be unique per instance"
    # Flag schema preserved
    assert "level" in a.event1.flags


def test_invalid_flag_type_raises():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    try:
        e.set(flags={"level": 123})  # wrong type
        assert False, "TypeError expected for wrong flag type"
    except TypeError:
        pass


def test_history_pruning():
    e = Event()
    e.max_history_time = 0.01
    e.set()
    assert len(e.history) >= 1
    # Force prune as-if lots of time passed
    now_future = time.monotonic() + 10.0
    e._prune_history(now=now_future)
    assert len(e.history) == 0, "History should be pruned when outside max_history_time window"


def test_wait_for_data_equals():
    e = Event()
    expected_payload = {"state": "ready", "count": 2}

    def setter():
        time.sleep(0.05)
        # Use a fresh dict to ensure we're testing equality, not identity
        e.set(data=dict(expected_payload))

    threading.Thread(target=setter, daemon=True).start()
    ok = e.wait(predicate=pred_data_equals(expected_payload), timeout=1.0)
    assert ok is True, "Event.wait with pred_data_equals should succeed when entire data matches"
    assert e.getData() == expected_payload, "Event data should equal the expected payload"


def test_counting_event_waits_for_value():
    e = Event()
    target_value = 7

    def counter():
        # Count up to the target, emitting an event at each step
        for i in range(target_value + 1):
            time.sleep(0.02)
            e.set(data=i)

    threading.Thread(target=counter, daemon=True).start()
    ok = e.wait(predicate=pred_data_equals(target_value), timeout=1.0)
    assert ok is True, "Waiter should trigger when the counter reaches the target value"
    assert e.getData() == target_value, "Event data should equal the target count when unblocked"


def test_listener_sees_snapshot_under_back_to_back_sets():
    """
    Repro the original race: emit val1 then val2 quickly.
    Listener predicate matches val1; it must receive the val1 payload,
    not the later overwritten value.
    """
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    seen = []
    hit = threading.Event()

    def cb(data):
        seen.append(data)
        hit.set()

    # Listener waits for level == "val1"
    listener = e.on(callback=cb,
                    predicate=pred_flag_equals("level", "val1"),
                    once=True,
                    input_data=True)

    def producer():
        e.set(data={"k": "v1"}, flags={"level": "val1"})
        # Immediately overwrite with another event
        e.set(data={"k": "v2"}, flags={"level": "val2"})

    threading.Thread(target=producer, daemon=True).start()

    assert hit.wait(1.0) is True, "Listener should fire"
    assert seen == [{"k": "v1"}], "Listener must receive the snapshot that matched (val1), not the overwrite"
    listener.stop()


def test_stale_precheck_returns_exact_snapshot_single_event():
    """
    Fire an event, then attach a waiter with stale_event_time.
    The waiter should trigger immediately and its snapshot should equal the payload at emit time.
    """
    e = Event(flags=[EventFlag(id="tag", data_type=str)])
    payload = {"x": 42}
    e.set(data=payload, flags={"tag": "past"})
    time.sleep(0.05)  # within the stale window below

    w = EventWaiter(events=e,
                    predicates=pred_flag_equals("tag", "past"),
                    stale_event_time=0.5,
                    timeout=0.5)

    start = time.monotonic()
    ok = w.wait()
    elapsed = time.monotonic() - start

    assert ok is True, "Stale precheck waiter should trigger immediately"
    assert elapsed < 0.1, "Should be nearly instantaneous"
    # matched snapshot should be present
    (flags, data) = w.matched_payloads[0]
    assert flags == {"tag": "past"}
    assert data == payload


def test_stale_precheck_multi_events_positions_filled_correctly():
    """
    Two events; we emit only one first, then attach a waiter for both with wait_for_all=True.
    Then we emit the second. Ensure each matched position contains its own snapshot.
    """
    e1 = Event(flags=[EventFlag(id="a", data_type=str)])
    e2 = Event(flags=[EventFlag(id="b", data_type=str)])

    p1 = {"p": 1}
    p2 = {"p": 2}

    # Fire e1 in the past
    e1.set(data=p1, flags={"a": "x"})
    time.sleep(0.05)  # inside stale window

    w = EventWaiter(events=[e1, e2],
                    predicates=[pred_flag_equals("a", "x"), pred_flag_equals("b", "y")],
                    wait_for_all=True,
                    stale_event_time=0.5,
                    timeout=0.5)

    # The stale precheck should mark position 0 true, position 1 false (until e2 fires)
    # Now fire e2 live
    def later():
        time.sleep(0.05)
        e2.set(data=p2, flags={"b": "y"})

    threading.Thread(target=later, daemon=True).start()

    ok = w.wait()
    assert ok is True
    assert w.events_finished == [True, True], "Both positions should be satisfied"

    (f1, d1) = w.matched_payloads[0]
    (f2, d2) = w.matched_payloads[1]
    assert f1 == {"a": "x"} and d1 == p1, "Position 0 should come from stale snapshot of e1"
    assert f2 == {"b": "y"} and d2 == p2, "Position 1 should come from live e2 dispatch"


def test_same_event_twice_different_predicates_snapshots_preserved():
    """
    Watch the SAME event twice, different predicates. Emit two distinct payloads;
    ensure each position captures the snapshot that matched its own predicate.
    """
    e = Event(flags=[EventFlag(id="mode", data_type=str)])
    p_auto = {"src": "first"}
    p_manual = {"src": "second"}

    w = EventWaiter(events=[e, e],
                    predicates=[pred_flag_equals("mode", "auto"),
                                pred_flag_equals("mode", "manual")],
                    wait_for_all=True,
                    timeout=1.0)

    def emit():
        time.sleep(0.02)
        e.set(data=p_auto, flags={"mode": "auto"})
        time.sleep(0.02)
        e.set(data=p_manual, flags={"mode": "manual"})

    threading.Thread(target=emit, daemon=True).start()
    ok = w.wait()
    assert ok is True
    (f0, d0) = w.matched_payloads[0]
    (f1, d1) = w.matched_payloads[1]
    assert f0 == {"mode": "auto"} and d0 == p_auto
    assert f1 == {"mode": "manual"} and d1 == p_manual


def test_listener_builds_from_matched_payloads_multi_event():
    """
    Listener over two events should deliver a dict mapping each Event -> its matched payload snapshot.
    """
    e1 = Event(flags=[EventFlag(id="kind", data_type=str)])
    e2 = Event()

    delivered = []
    hit = threading.Event()

    def cb(data):
        # data: {e1: payload1, e2: payload2}
        delivered.append(data)
        hit.set()

    listener = EventListener(event=[e1, e2],
                             predicate=[pred_flag_equals("kind", "alpha"), None],
                             callback=cb,
                             once=True,
                             input_data=True,
                             wait_for_all=True)

    listener.start()

    def emitter():
        e1.set(data={"v": "A"}, flags={"kind": "alpha"})
        e2.set(data={"v": "B"})

    threading.Thread(target=emitter, daemon=True).start()
    assert hit.wait(1.0) is True
    assert len(delivered) == 1
    out = delivered[0]
    assert out[e1] == {"v": "A"}
    assert out[e2] == {"v": "B"}
    listener.stop()


def test_copy_data_on_set_true_isolation_from_mutation():
    """
    With copy_data_on_set=True (default), mutating the producer's dict after set()
    must NOT affect the stored payload nor the snapshot seen by listeners.
    """
    e = Event(copy_data_on_set=True)
    src = {"k": ["a", "b"]}
    e.set(data=src)
    src["k"].append("c")  # mutate after set

    # Access via getData(copy=False) to inspect stored payload
    stored = e.getData(copy=False)
    assert stored == {"k": ["a", "b"]}, "Stored payload should be isolated by deepcopy"


def test_copy_data_on_set_false_allows_aliasing():
    """
    With copy_data_on_set=False, we alias the input. Mutating after set() will reflect
    in the stored payload. (Useful test so behavior is explicit.)
    """
    e = Event(copy_data_on_set=False)
    src = {"k": ["a"]}
    e.set(data=src)
    src["k"].append("b")
    stored = e.getData(copy=False)
    assert stored == {"k": ["a", "b"]}, "Aliasing expected when copy_data_on_set=False"


def test_history_used_in_has_match_in_window_uses_stored_data_not_latest():
    """
    Ensure has_match_in_window evaluates predicate against the data from history,
    not the latest event.data.
    """
    e = Event(flags=[EventFlag(id="lvl", data_type=str)])
    e.set(data={"n": 1}, flags={"lvl": "x"})
    time.sleep(0.02)
    # overwrite with non-matching data
    e.set(data={"n": 2}, flags={"lvl": "z"})

    # Ask for a stale window covering the first event; predicate checks for n == 1
    pred = lambda f, d: d.get("n") == 1 and f.get("lvl") == "x"
    ok = e.has_match_in_window(pred, window=0.5)
    assert ok is True, "Should match against stored snapshot in history even if latest is different"


if __name__ == '__main__':
    # Execute all test functions from above here
    test_wait_for_data_equals()
    test_counting_event_waits_for_value()
    test_listener_sees_snapshot_under_back_to_back_sets()
    test_stale_precheck_returns_exact_snapshot_single_event()
    test_stale_precheck_multi_events_positions_filled_correctly()
    test_same_event_twice_different_predicates_snapshots_preserved()
    test_listener_builds_from_matched_payloads_multi_event()
    test_copy_data_on_set_true_isolation_from_mutation()
    test_copy_data_on_set_false_allows_aliasing()
    test_history_used_in_has_match_in_window_uses_stored_data_not_latest()
    test_event_definition_decorator()


