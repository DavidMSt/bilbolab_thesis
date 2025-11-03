from __future__ import annotations

import enum
import queue
import threading
from copy import deepcopy
import time
from dataclasses import is_dataclass, dataclass
from typing import Callable, Any, Optional, Union, Literal, TypeAlias
from collections import deque
import weakref
import fnmatch

# === CUSTOM MODULES ===================================================================================================
from core.utils.callbacks import callback_definition, CallbackContainer, Callback
from core.utils.dataclass_utils import deepcopy_dataclass
from core.utils.dict_utils import optimized_deepcopy
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from core.utils.singleton import _SingletonMeta
from core.utils.time import setTimeout, setInterval
from core.utils.uuid_utils import generate_uuid

# === GLOBAL VARIABLES =================================================================================================
logger = Logger('events')


# === HELPERS ==========================================================================================================
def _check_id(id: str):
    # Check that the id does not contain any special characters except for "_"
    if not isinstance(id, str):
        raise TypeError("Event ID must be a string")
    if not id:
        raise ValueError("Event ID cannot be empty")
    if not all(c.isalnum() or c == '_' for c in id):
        raise ValueError("Event ID can only contain alphanumeric characters and underscores")
    return True


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
    id: str

    data_type: type | None
    flags: dict[str, EventFlag]
    copy_data_on_set = True
    data_is_static_dict: bool
    custom_data_copy_function = None
    max_history_time: float = 10.0  # Seconds

    parent: EventContainer | None = None
    data: Any
    callbacks: EventCallbacks
    history: deque[tuple[float, dict[str, Any], Any]]
    dict_copy_cache = None

    # === INIT =========================================================================================================
    def __init__(self,
                 id: str = None,
                 data_type: type | None = None,
                 flags: EventFlag | list[EventFlag] = None,
                 copy_data_on_set: bool = True,
                 data_is_static_dict: bool = False, ):

        if id is None:
            id = generate_uuid()

        _check_id(id)

        self.id = id

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

    # === PROPERTIES ===================================================================================================
    @property
    def uid(self) -> str:
        if self.parent is None:
            return self.id
        else:
            if self.parent.id is not None:
                return f"{self.parent.id}:{self.id}"
            else:
                return self.id

    # === METHODS ======================================================================================================
    def on(self,
           callback: Callback | Callable,
           predicate: Predicate = None,
           once: bool = False,
           stale_event_time=None,
           timeout=None,
           input_data=True,
           max_rate=None) -> SubscriberListener:
        """
        Always return a SubscriberListener as the handle.
        If once=True, the underlying Subscriber is once=True and the listener will
        auto-stop after the first delivery.
        """
        sub = Subscriber(
            id=f"{self.uid}_subscriber",
            events=(self, predicate) if predicate else self,
            once=once,
            stale_event_time=stale_event_time,
        )

        listener = SubscriberListener(
            subscriber=sub,
            callback=callback,
            input_data=input_data,
            pass_only_data=True,  # pass only .data by default (like old listeners)
            max_rate=max_rate,
            spawn_new_threads=False,  # mimic old default; flip if you want
            auto_stop_on_first=once,  # NEW: stop listener after first emit when once=True
            timeout=timeout,
        )
        listener.start()
        return listener

    # ------------------------------------------------------------------------------------------------------------------
    def wait(self, predicate: Predicate = None, timeout: float = None,
             stale_event_time: float = None) -> SubscriberMatch | None:
        subscriber = Subscriber(events=(self, predicate),
                                timeout=timeout,
                                stale_event_time=stale_event_time,
                                once=True,
                                )
        return subscriber.wait()

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
    def get_data(self, copy: bool = True) -> Any:
        if copy:
            return self._copy_payload(self.data)
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
            logger.warning(f"Could not copy data for event {self}: {e}. Subsequent set() calls will not copy.")

        return payload

    # ------------------------------------------------------------------------------------------------------------------`
    def __repr__(self):
        return f"<Event {self.uid}>"


# === SUBSCRIBER =======================================================================================================
@callback_definition
class SubscriberCallbacks:
    finished: CallbackContainer
    timeout: CallbackContainer


@dataclass
class SubscriberMatch:
    time: float

    eventspec: str  # This is the event spec for the match that triggered
    match: str | list[str]  # These are event ids that triggered the match


    match_data: Any | dict[str, Any]
    data: Any | dict[str, Any]  # Either the data for one event or for multiple events in a dict
    flags: Any | dict[str, Any]  # Either the flags for one event or for multiple events in a dict


class SubscriberType(enum.StrEnum):
    AND = "AND"
    OR = "OR"


# Type definition for event specifications
EventSpec: TypeAlias = Union[
    str,  # Event ID
    Event,  # Single event
    tuple[str, Predicate],
    tuple[Event, Predicate],  # Event with predicate
    'Subscriber',  # Nested subscriber
    list[Union[  # List of any combination
        str,
        Event,
        tuple[Event, Predicate],
        tuple[str, Predicate],
        'Subscriber'
    ]]
]


class Subscriber:
    id: str

    events: list[Event | Subscriber]
    predicates: list[Predicate | None] | None

    timeout: float | None
    stale_event_time: float | None
    once: bool
    type: SubscriberType

    finished_events: dict[str, bool]
    payloads: dict[str, dict]

    matches: list[SubscriberMatch]  # Previous matches (recent, pruned)

    _save_matches: bool
    _match_save_time: float
    _abort: bool

    # fan-out support
    _wait_queues: set[queue.Queue]
    _wait_queue_maxsize: int
    _wq_lock: threading.RLock

    _event_loop: EventLoop | None = None
    _SENTINEL = object()

    def __init__(self,
                 events: EventSpec,
                 id: str | None = None,
                 type: SubscriberType | None = SubscriberType.AND,
                 timeout: float | None = None,
                 stale_event_time: float | None = None,

                 once: bool = False,
                 callback: Callable | Callback | None = None,
                 execute_callback_in_thread: bool = True,

                 save_matches: bool = True,
                 match_save_time: float | None = 10,
                 queue_maxsize: int = 1,
                 event_loop: EventLoop | None = None):

        if id is None:
            id = generate_uuid(prefix="subscriber_")

        self.id = id

        self.callbacks = SubscriberCallbacks()

        if not isinstance(events, list):
            events = [events]

        self.events = []
        self.predicates = []

        for eventspec in events:
            if isinstance(eventspec, str):
                pattern_subscriber = PatternSubscriber(id=eventspec, pattern=eventspec)
                self.events.append(pattern_subscriber)
                self.predicates.append(None)
                pattern_subscriber.callbacks.finished.register(self._child_subscriber_callback,
                                                               inputs={'subscriber': pattern_subscriber})
            elif isinstance(eventspec, tuple) and len(eventspec) == 2 and isinstance(eventspec[0], str) and callable(
                    eventspec[1]):
                pattern: str = eventspec[0]  # type: ignore
                predicate: Callable = eventspec[1]
                pattern_subscriber = PatternSubscriber(pattern=pattern, predicate=predicate)
                self.events.append(pattern_subscriber)
                self.predicates.append(None)
                pattern_subscriber.callbacks.finished.register(self._child_subscriber_callback,
                                                               inputs={'subscriber': pattern_subscriber})
            elif isinstance(eventspec, Event):
                self.events.append(eventspec)
                self.predicates.append(None)
            elif isinstance(eventspec, tuple) and len(eventspec) == 2 and isinstance(eventspec[0], Event) and callable(
                    eventspec[1]):
                ev, pr = eventspec
                self.events.append(ev)
                self.predicates.append(pr)
            elif isinstance(eventspec, Subscriber):
                self.events.append(eventspec)
                self.predicates.append(None)
                eventspec.callbacks.finished.register(self._child_subscriber_callback,
                                                      inputs={'subscriber': eventspec})

        self.timeout = timeout
        self.stale_event_time = stale_event_time
        self.type = type
        self.once = once

        self.finished_events = {event.uid: False for event in self.events}
        self.payloads = {event.uid: {'data': None, 'flags': None, 'match_data': None} for event in self.events}

        self.logger = Logger(f"Subscriber ({[event.uid for event in self.events]})", "DEBUG")

        self._abort = False
        self._wait_queues = set()
        self._wait_queue_maxsize = max(0, queue_maxsize)
        self._wq_lock = threading.RLock()

        if event_loop is None:
            event_loop = active_event_loop
        self._event_loop = event_loop

        if callback is not None:
            self.callbacks.finished.register(callback)

        self._save_matches = save_matches
        self._match_save_time = match_save_time
        self.execute_callback_in_thread = execute_callback_in_thread
        self.matches = []

        self.logger.debug("Add to event loop")

        self._event_loop.add_subscriber(self)
        self._check_child_subscribers()

    # === PROPERTIES ===================================================================================================
    @property
    def uid(self):
        return self.id

    # === METHODS ======================================================================================================
    def wait(self,
             timeout: float | None = None,
             stale_event_time: float | None = None) -> SubscriberMatch | None:
        """
        Block until a new match arrives (fan-out via per-waiter queue).
        If a recent match exists within the provided or configured stale window,
        return it immediately without blocking.

        Returns:
            SubscriberMatch if available, else None on timeout/stop.
        """
        self.logger.debug("Wait")
        if timeout is None:
            timeout = self.timeout

        # 1) Fast path: return a recent match within the stale window (if requested)
        window = stale_event_time if stale_event_time is not None else self.stale_event_time
        if window and window > 0:
            now = time.monotonic()
            cutoff = now - window
            # Read-mostly; fine to snapshot without a separate lock
            recent = None
            for m in reversed(self.matches):
                if m.time >= cutoff:
                    recent = m
                    break
                # matches are time-ordered; we can break once older than cutoff
                # but only if we know they are sorted; they are appended in _fire() -> yes
            if recent is not None:
                return recent

        # 2) Slow path: set up a one-off queue and block until push or timeout
        q = queue.Queue(maxsize=self._wait_queue_maxsize)
        with self._wq_lock:
            # if stop() already called, don't register / return None
            if self._abort:
                return None
            self._wait_queues.add(q)

        try:
            try:
                match = q.get(timeout=timeout) if timeout is not None else q.get()
            except queue.Empty:
                self.callbacks.timeout.call()
                return None

            if match is self._SENTINEL:
                return None
            return match
        finally:
            with self._wq_lock:
                self._wait_queues.discard(q)

    # ------------------------------------------------------------------------------------------------------------------
    def on(self,
           callback: Callback | Callable,
           once: bool = False,
           timeout=None,
           input_data=True,
           pass_only_data=True,
           max_rate=None) -> SubscriberListener:
        """
        Always return a SubscriberListener as the handle.
        If once=True, the underlying Subscriber is once=True and the listener will
        auto-stop after the first delivery.
        """

        listener = SubscriberListener(
            subscriber=self,
            callback=callback,
            input_data=input_data,
            pass_only_data=pass_only_data,  # pass only .data by default (like old listeners)
            max_rate=max_rate,
            spawn_new_threads=False,  # mimic old default; flip if you want
            auto_stop_on_first=once,  # NEW: stop listener after first emit when once=True
            timeout=timeout,
        )
        listener.start()
        return listener

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        """Abort future waiting, deregister from loop, and wake any blocking wait()."""
        self._abort = True
        self._unsubscribe()
        # Broadcast sentinel to all registered wait queues
        with self._wq_lock:
            for q in list(self._wait_queues):
                self._nonblocking_push(q, self._SENTINEL)

    # === PRIVATE METHODS ==============================================================================================
    def set_match(self, event_id: str, flags: dict[str, Any] | None, data: Any):

        event = self._get_event_by_id(event_id)
        if event is None:
            self.logger.error(f"Event {event_id} not found. Ignoring.")
            return

        self.payloads[event_id]['flags'] = flags

        if isinstance(event, Subscriber):
            self.payloads[event_id]['match_data'] = data
            self.payloads[event_id]['data'] = data.data
        else:
            self.payloads[event_id]['match_data'] = data
            self.payloads[event_id]['data'] = data

        self.finished_events[event_id] = True

        self.logger.debug(f"Match: {event_id}, flags: {flags}, data: {data}")

        if self.type == SubscriberType.AND:
            satisfied = all(self.finished_events.values())
        else:
            satisfied = any(self.finished_events.values())

        if satisfied:
            self._fire()

    # ------------------------------------------------------------------------------------------------------------------
    def _fire(self):
        self.logger.debug("Subscriber satisfied. Gathering data and flags.")

        # Gather matched data/flags
        if len(self.events) == 1:
            matched_event = self.events[0].uid
            data = self.payloads[self.events[0].uid]['data']
            match_data = self.payloads[self.events[0].uid]['data']
            flags = self.payloads[self.events[0].uid]['flags']
        else:
            if self.type == SubscriberType.AND:
                matched_event = [event.uid for event in self.events]
            else:
                matched_event = next((event for event, finished in self.finished_events.items() if finished), None)

            data = {event_id: payload['data'] for event_id, payload in self.payloads.items()}
            match_data = {event_id: payload['match_data'] for event_id, payload in self.payloads.items()}
            flags = {event_id: payload['flags'] for event_id, payload in self.payloads.items()}

        match = SubscriberMatch(
            time=time.monotonic(),
            match=matched_event,
            data=data,
            match_data=match_data,
            flags=flags
        )

        self.logger.debug(f"Match: {matched_event}. Data: {match}")

        # Once semantics: prevent future matches and unsubscribe from loop
        if self.once:
            self._abort = True
            self._unsubscribe()

        # Save for stale-window replay and prune old ones
        if self._save_matches:
            self.matches.append(match)
            self._prune_matches()

        # Reset state for continuous subscribers
        if not self.once:
            self.finished_events = {event.uid: False for event in self.events}
            self.payloads = {event.uid: {'data': None, 'match_data': None, 'flags': None} for event in self.events}

        # Broadcast to all current waiters (non-blocking, drop-oldest)
        with self._wq_lock:
            for q in list(self._wait_queues):
                self._nonblocking_push(q, match)

        self.logger.debug("Subscriber satisfied. Fire callback(s)")
        self._execute_callback(match,
                               execute_callback_in_thread=self.execute_callback_in_thread,
                               input_match_data=True)

    # ------------------------------------------------------------------------------------------------------------------
    def _nonblocking_push(self, q: queue.Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            # drop oldest item to make room
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                # still full -> give up; likely abandoned waiter
                pass

    # ------------------------------------------------------------------------------------------------------------------
    def _execute_callback(self, match_data, input_match_data: bool = True, execute_callback_in_thread: bool = True):
        for callback in self.callbacks.finished.callbacks:
            if execute_callback_in_thread:
                if input_match_data:
                    threading.Thread(target=callback, args=(match_data,), daemon=True).start()
                else:
                    threading.Thread(target=callback, args=(), daemon=True).start()
            else:
                if input_match_data:
                    callback(match_data)
                else:
                    callback()

    # ------------------------------------------------------------------------------------------------------------------
    def _unsubscribe(self):
        if self._event_loop is not None:
            self._event_loop.remove_subscriber(self)

    # ------------------------------------------------------------------------------------------------------------------
    def _child_subscriber_callback(self, data, subscriber: Subscriber = None):
        self.logger.debug(f"Child subscriber callback. Child: {subscriber.__repr__()}")
        self.set_match(subscriber.id, None, data)

    # ------------------------------------------------------------------------------------------------------------------
    def _prune_matches(self):
        """Remove matches older than match_save_time from the current time"""
        if not self._match_save_time:
            return

        now = time.monotonic()
        cutoff = now - self._match_save_time
        # matches are appended in chronological order; prune from the front
        self.matches = [match for match in self.matches if match.time >= cutoff]

    # ------------------------------------------------------------------------------------------------------------------
    def _check_child_subscribers(self):
        if not self.stale_event_time:
            return

        child_subscribers = [event for event in self.events if isinstance(event, Subscriber)]
        now = time.monotonic()
        cutoff = now - self.stale_event_time

        for child_subscriber in child_subscribers:
            # Find the most recent match within the stale window
            recent_matches = [m for m in child_subscriber.matches if m.time >= cutoff]
            if recent_matches:
                latest_match = max(recent_matches, key=lambda m: m.time)
                self.set_match(child_subscriber.id, latest_match.flags, latest_match.data)

    # ------------------------------------------------------------------------------------------------------------------
    def _get_event_by_id(self, event_id: str) -> Event | None:
        for event in self.events:
            if event.uid == event_id:
                return event
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        return f"<Subscriber {self.id} {[event.__repr__() for event in self.events]}>"


# === Pattern Subscriber ===============================================================================================
class PatternSubscriber(Subscriber):
    """
        Pattern syntax (fnmatch) — NOT regex

    PatternSubscriber uses Python’s `fnmatch` “shell-style” wildcards to match an event’s UID.
    This is the same family of patterns you’d use in a Unix shell (e.g., `ls *.txt`), NOT
    regular expressions. Internally we use `fnmatch.fnmatchcase`, so matching is always
    case-sensitive and consistent across platforms.

    Special characters
    ------------------
    *  (asterisk)      Matches any sequence of characters (including empty).
                       Examples:
                         "*:ready"       → any UID that ends with ":ready"
                         "sensor:*"      → any UID that starts with "sensor:"
                         "*"             → everything

    ?  (question mark) Matches any single character.
                       Examples:
                         "cam_?"         → "cam_1", "cam_A", but not "cam_10"
                         "room_1?:temp"  → "room_10:temp", "room_1A:temp"

    [...] (char class) Matches exactly one character from the set/range.
                       Examples:
                         "room_[1-9]"    → "room_1" … "room_9"
                         "room_[0-9][0-9]" → "room_00" … "room_99"
                         "cam_[ab]"      → "cam_a" or "cam_b"

    [!...] (negation)  Matches exactly one character NOT in the set/range.
                       Example:
                         "node_[!x]_*"   → UIDs where the character after "node_" is anything but "x"

    Literal specials
    ----------------
    To match a literal special character, wrap it in a character class:

      "[*]"  matches a literal asterisk "*"
      "[?]"  matches a literal question mark "?"
      "[[]", "[]]" match literal "[" and "]" respectively
      "[-]"  matches a literal dash inside a char class (otherwise dash defines ranges)

    Notes & gotchas
    ---------------
    • Not regex: constructs like "(...)", "|", "+", "{2,3}", lookaheads, etc. are NOT supported.
      If you need regex power, use `re` yourself.

    • "**" has no special recursive meaning here; "**" is just two "*" characters.

    • Leading dots: like Unix shells, a pattern that doesn’t include a dot at the beginning
      won’t match names that start with ".". In practice that means "*" does NOT match ".hidden"
      unless your pattern explicitly accounts for it (e.g., ".*" or ".*:ready").

    • Path separators are not special: matching is done against the raw UID string
      (e.g., "parent:child"), so ":" is just another character.

    • Case sensitivity: we use `fnmatch.fnmatchcase`, so "A" ≠ "a" on all platforms.

    Handy examples for UIDs
    -----------------------
      "*:ready"             → any event whose UID ends with ":ready"
      "sensor_*"            → events starting with "sensor_"
      "room_[12][0-9]:temp" → "room_10:temp" … "room_29:temp"
      "camera_[!x]*"        → UIDs like "camera_a1", "camera_Z", but not "camera_x…"
      "*event1"             → anything ending with "event1"
      "parent:*:done"       → a three-segment UID that ends with ":done"

    Tips
    ----
    • You can inspect how `fnmatch` compiles a pattern by calling `fnmatch.translate(pattern)`,
      which returns the equivalent regex—useful for debugging—but remember your input is still
      shell-style, not regex.

    • In this event bus, the pattern is applied to `Event.uid` (e.g., "parent:child").
      Choose your UID scheme with that in mind.
    """
    pattern: str
    global_predicate: Predicate | None = None

    def __init__(self,
                 pattern: str,
                 *,
                 predicate: Predicate | None = None,
                 id: str | None = None,
                 **kwargs):
        # Force OR semantics for pattern subscribers
        if 'type' in kwargs and kwargs['type'] != SubscriberType.OR:
            raise ValueError("PatternSubscriber always uses OR semantics.")
        kwargs['type'] = SubscriberType.OR

        # Give it a readable id if none provided
        if id is None:
            id = f"pattern:{pattern}"

        # Stash pattern & global predicate
        self.pattern = pattern
        self.global_predicate = predicate

        # Important: start with *no* concrete events. We’ll get them from EventLoop.
        super().__init__(events=[], id=id, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    def add_event(self, event: Event):
        if event in self.events:
            return

        self.logger.debug(f"Add event: {event.uid} to pattern {self.pattern}")
        self.events.append(event)
        self.predicates.append(self.global_predicate)

        self.finished_events[event.uid] = False
        self.payloads[event.uid] = {'data': None, 'flags': None}

        # Ensure we receive event->subscriber dispatch even if add_event() is called directly
        active_event_loop.subscribers_by_event.setdefault(event, set()).add(self)

        if self.stale_event_time and self.stale_event_time > 0:
            now = time.monotonic()
            if getattr(event, "max_history_time", 0) < self.stale_event_time:
                event.max_history_time = self.stale_event_time
            match = event.first_match_in_window(self.global_predicate, self.stale_event_time, now=now)
            if match is not None:
                flags, data = match
                self.set_match(event.uid, flags, data)

    # ------------------------------------------------------------------------------------------------------------------
    def remove_event(self, event: Event):
        """Detach a concrete Event from this PatternSubscriber."""
        if event not in self.events:
            return

        # Remove mappings
        idx = self.events.index(event)
        self.events.pop(idx)

        # Predicates mirror events list order
        if self.predicates and idx < len(self.predicates):
            self.predicates.pop(idx)

        # Drop state keyed by uid
        self.finished_events.pop(event.uid, None)
        self.payloads.pop(event.uid, None)

        # >>> IMPORTANT: remove from the reverse index in the event loop
        loop = active_event_loop  # or self._event_loop if assigned
        subs = loop.subscribers_by_event.get(event)
        if subs:
            subs.discard(self)
            if not subs:
                # Optional: clean empty sets
                loop.subscribers_by_event.pop(event, None)

    def __repr__(self):
        return f"<PatternSubscriber {self.id} pattern={self.pattern} events={[e.uid for e in self.events]}>"


# === SUBSCRIBER LISTENER ==============================================================================================
class SubscriberListener:
    _max_rate: float | None
    _spawn_new_threads: bool
    _input_data: bool  # Whether to pass the data to the callback
    _pass_only_data: bool  # Whether to strip the data from the match class
    _timeout: float | None
    _exit: bool = False
    _thread: threading.Thread | None = None

    _auto_stop_on_first: bool = False

    _last_callback_time: float | None = None

    _stop_event: Event

    def __init__(self,
                 subscriber: Subscriber,
                 callback: Callable | Callback,
                 input_data: bool = True,
                 pass_only_data: bool = True,
                 max_rate: float | None = None,
                 spawn_new_threads: bool = False,
                 auto_stop_on_first: bool = False,
                 timeout: float | None = None, ):

        self.subscriber = subscriber
        self.callback = callback
        self._input_data = input_data
        self._pass_only_data = pass_only_data
        self._max_rate = max_rate
        self._spawn_new_threads = spawn_new_threads
        self._auto_stop_on_first = auto_stop_on_first
        self._timeout = timeout

        self.logger = Logger(f"Subscriber {self.subscriber.id} listener", "DEBUG")

        self._stop_event = Event(id=f"{id(self)}_stop")
        # self._fire_event = Event(id=f"{id(self)}_fire")

        self._compound_subscriber = Subscriber(
            events=[
                self.subscriber,
                self._stop_event,
            ],
            type=SubscriberType.OR,
        )

        register_exit_callback(self.stop)

    # === METHODS ======================================================================================================
    def start(self):
        self._thread = threading.Thread(target=self._task, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self, *args, **kwargs):
        """Public stop: signal + join if called from another thread."""
        self._request_stop()
        # Only join if we're NOT on the worker thread
        if self._thread is not None and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join()

    # === PRIVATE METHODS ==============================================================================================
    def _task(self):
        while not self._exit:
            result = self._compound_subscriber.wait(timeout=self._timeout)

            if result is None:
                self.logger.warning("Subscriber wait time out. Not implemented yet")
                self.stop()
                continue

            if self._exit:
                break

            # Check if it is the stop event
            if result.match == self._stop_event.uid:
                self.logger.debug("Received stop event. Stopping.")
                self._request_stop()
                break

            if self._max_rate:
                now = time.monotonic()
                min_interval = 1.0 / self._max_rate

                # If we've fired before and the interval not met → drop this event
                if self._last_callback_time is not None and (now - self._last_callback_time) < min_interval:
                    continue  # just skip this event

                # Update the timestamp for the allowed callback
                self._last_callback_time = now

            if self._pass_only_data:
                data = result.data[self.subscriber.id]
            else:
                # TODO: Here I need to remove the stop event from the Match
                data = result

            self._execute_callback(data, input_data=self._input_data, spawn_thread=self._spawn_new_threads)

            if self._auto_stop_on_first:
                self._request_stop()

    # ------------------------------------------------------------------------------------------------------------------
    def _execute_callback(self, data, input_data, spawn_thread):
        if spawn_thread:
            if input_data:
                threading.Thread(target=self.callback, args=(data,), daemon=True).start()
            else:
                threading.Thread(target=self.callback, args=(), daemon=True).start()
        else:
            if input_data:
                self.callback(data)
            else:
                self.callback()

    # ------------------------------------------------------------------------------------------------------------------
    def _request_stop(self):
        """Signal the thread to exit, without joining (safe to call from worker)."""
        self._exit = True
        try:
            self._stop_event.set()
        except Exception:
            pass  # safe: may already be stopping


# === EVENT LOOP =======================================================================================================
class EventLoop(metaclass=_SingletonMeta):
    """
    Scalable event dispatcher with:
      - per-waiter queues (push on match)
      - stale-window precheck on registration
      - exact snapshot delivery
    """
    subscribers: list[Subscriber]
    events: weakref.WeakSet

    def __init__(self):
        self.events = weakref.WeakSet()
        self.subscribers: list[Subscriber] = []

        self.subscribers_by_event: dict[Event, set[Subscriber]] = {}
        self._subscribers_lock = threading.RLock()
        self._pattern_subscribers: set[PatternSubscriber] = set()

    # ------------------------------------------------------------------------------------------------------------------
    def addEvent(self, event: Event):
        with self._subscribers_lock:
            if event in self.events:
                return
            self.events.add(event)
            event.callbacks.set.register(Callback(
                function=self._event_set,
                inputs={'event': event}
            ))

            # NEW: attach to any pattern subscribers that match this event's UID
            for ps in list(self._pattern_subscribers):
                self._attach_pattern_if_match(ps, event)

    # ------------------------------------------------------------------------------------------------------------------
    def add_subscriber(self, subscriber: Subscriber):
        with self._subscribers_lock:
            self.subscribers.append(subscriber)

            if isinstance(subscriber, PatternSubscriber):
                # Track pattern subscribers for future events
                self._pattern_subscribers.add(subscriber)
                # Attach all existing events that match
                for ev in list(self.events):
                    self._attach_pattern_if_match(subscriber, ev)
            else:
                # Normal (non-pattern) subscribers: wire their explicit events
                for ev in subscriber.events:
                    self.subscribers_by_event.setdefault(ev, set()).add(subscriber)

            # Ensure history covers the stale window for *currently attached* events
            if subscriber.stale_event_time and subscriber.stale_event_time > 0:
                now = time.monotonic()
                for i, ev in enumerate(subscriber.events):
                    if isinstance(ev, Subscriber):  # nested subscribers handled elsewhere
                        continue
                    if getattr(ev, "max_history_time", 0) < subscriber.stale_event_time:
                        ev.max_history_time = subscriber.stale_event_time
                    pred = (subscriber.predicates[i]
                            if (subscriber.predicates and i < len(subscriber.predicates)) else None)
                    match = ev.first_match_in_window(pred, subscriber.stale_event_time, now=now)
                    if match is not None:
                        flags, data = match
                        subscriber.set_match(ev.uid, flags, data)

    # ------------------------------------------------------------------------------------------------------------------
    def remove_subscriber(self, waiter: Subscriber):
        with self._subscribers_lock:
            self._unsafe_removeWaiter(waiter)

    # ------------------------------------------------------------------------------------------------------------------
    def _unsafe_removeWaiter(self, waiter: Subscriber):
        if waiter in self.subscribers:
            self.subscribers.remove(waiter)
        for ev in getattr(waiter, "events", ()):
            s = self.subscribers_by_event.get(ev)
            if s is not None:
                s.discard(waiter)
                if not s:
                    self.subscribers_by_event.pop(ev, None)
        # Also drop from the pattern registry
        if isinstance(waiter, PatternSubscriber):
            self._pattern_subscribers.discard(waiter)

    # ------------------------------------------------------------------------------------------------------------------
    def _event_set(self, data, event: Event, flags, *args, **kwargs):
        with self._subscribers_lock:
            watchers = list(self.subscribers_by_event.get(event, ()))
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
                        if not waiter.finished_events[ev.uid]:
                            # waiter.events_finished[ev.id] = True
                            waiter.set_match(ev.uid, flags, data)

                if not matched_any_pos:
                    continue

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_pattern(event_uid: str, pattern: str) -> bool:
        # Support glob wildcards against full UID. Exact match if no wildcards.
        return fnmatch.fnmatchcase(event_uid, pattern)

    # ------------------------------------------------------------------------------------------------------------------
    def _attach_pattern_if_match(self, ps: PatternSubscriber, ev: Event):
        if self._check_pattern(ev.uid, ps.pattern):
            # Let the PatternSubscriber attach and sync its internal state
            ps.add_event(ev)
            # And also register it in the reverse index so it gets _event_set callbacks
            self.subscribers_by_event.setdefault(ev, set()).add(ps)


# === EVENT CONTAINER AND DECORATOR ====================================================================================
class EventContainer:
    id: str | None = None
    events: dict[str, Event]

    def __init__(self, id: str = None):
        _check_id(id)
        self.id = id
        self.events = {}

    def add_event(self, event: Event):
        if event.id in self.events:
            raise ValueError(f"Event {event.id} already exists.")
        if event.parent is not None:
            raise ValueError(f"Event {event.id} already has a parent.")
        self.events[event.id] = event
        event.parent = self


# === EVENT CONTAINER DECORATOR ========================================================================================
def event_definition(cls):
    """
    Per-instance Event fields that are automatically added as children of the container.

    Usage:
        @event_definition
        class RobotEvents(EventContainer):
            ready: Event = Event(flags=[EventFlag('level', str)])
            moved: Event

    Notes:
      - If the class does not subclass EventContainer, this decorator will still create
        `self.events` and inject a compatible `add_event` method on the instance.
      - Each instance receives fresh Event objects.
      - Each Event's `id` defaults to the attribute name (even when cloning a class-level template).
    """
    import sys
    import types as _types
    import typing
    from typing import get_origin, get_args

    original_init = getattr(cls, "__init__", None)

    # --- Resolve annotations (supports `from __future__ import annotations`) ---
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
        # Union handling (typing.Union or PEP 604)
        try:
            is_union = (origin is typing.Union) or (origin is _types.UnionType)
        except Exception:
            is_union = (origin is typing.Union)
        if is_union:
            return any(_is_event_type(arg) for arg in get_args(t))
        if origin is typing.ClassVar:
            return False
        return False

    def _clone_event_template(template: Event, new_id: str) -> Event:
        # Rebuild flags schema
        flags = [EventFlag(ef.id, ef.types) for ef in template.flags.values()]
        clone = Event(
            id=new_id,
            data_type=template.data_type,
            flags=flags,
            copy_data_on_set=template.copy_data_on_set,
            data_is_static_dict=template.data_is_static_dict,
        )
        # Copy non-ctor attributes
        clone.custom_data_copy_function = template.custom_data_copy_function
        clone.max_history_time = template.max_history_time
        return clone

    def _ensure_container_bits(self):
        # Ensure this instance has a place to register events
        if not hasattr(self, "events") or not isinstance(getattr(self, "events"), dict):
            self.events = {}

        # Provide add_event if missing (mirror EventContainer.add_event semantics)
        if not hasattr(self, "add_event") or not callable(getattr(self, "add_event")):
            def _add_event(_self, event: Event):
                if event.id in _self.events:
                    raise ValueError(f"Event {event.id} already exists.")
                if event.parent is not None:
                    raise ValueError(f"Event {event.id} already has a parent.")
                _self.events[event.id] = event
                event.parent = _self

            # Bind as a method on the instance
            setattr(self, "add_event", _add_event.__get__(self, self.__class__))

    def new_init(self, *args, **kwargs):
        # Run any user-defined __init__ first
        if original_init:
            original_init(self, *args, **kwargs)

        # Ensure container plumbing exists
        _ensure_container_bits(self)

        # 1) Process annotated attributes
        if isinstance(hints, dict):
            for attr_name, anno in hints.items():
                default_val = getattr(cls, attr_name, None)

                # If class-level default is an Event → clone it per instance and rename to attr_name
                if isinstance(default_val, Event):
                    ev = _clone_event_template(default_val, new_id=attr_name)
                    setattr(self, attr_name, ev)
                    # Register as child
                    self.add_event(ev)
                    continue

                # If annotated as Event (or Optional/Union including Event) and not set in instance → create fresh
                if _is_event_type(anno) and attr_name not in self.__dict__:
                    ev = Event(id=attr_name)
                    setattr(self, attr_name, ev)
                    self.add_event(ev)

        # 2) Pick up any unannotated class-level Event defaults
        for attr_name, value in vars(cls).items():
            if isinstance(value, Event) and attr_name not in self.__dict__:
                ev = _clone_event_template(value, new_id=attr_name)
                setattr(self, attr_name, ev)
                self.add_event(ev)

    cls.__init__ = new_init
    return cls


# ======================================================================================================================
active_event_loop = EventLoop()


# === TESTS ============================================================================================================
def test_event_wait_for_flag_predicate_new():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    hit = []

    def setter():
        time.sleep(0.05)
        e.set(data={"v": 1}, flags={"level": "high"})

    threading.Thread(target=setter, daemon=True).start()
    res = e.wait(predicate=pred_flag_equals("level", "high"), timeout=1.0)
    assert res is not None, "Event.wait should return a SubscriberMatch on success"
    assert e.get_data() == {"v": 1}


def test_event_wait_for_data_predicate_new():
    e = Event()

    def setter():
        time.sleep(0.05)
        e.set(data={"state": "ready"})

    threading.Thread(target=setter, daemon=True).start()
    res = e.wait(predicate=pred_data_dict_key_equals("state", "ready"), timeout=1.0)
    assert res is not None


def test_stale_event_wait_success_new():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    e.set(data={"x": 1}, flags={"level": "high"})
    time.sleep(0.2)
    res = e.wait(predicate=pred_flag_equals("level", "high"),
                 stale_event_time=1.0, timeout=0.2)
    assert res is not None, "Stale-window wait should return immediately for recent event"


def test_stale_event_wait_timeout_new():
    e = Event(flags=[EventFlag(id="level", data_type=str)])
    e.set(data={"x": 1}, flags={"level": "high"})
    time.sleep(0.3)
    res = e.wait(predicate=pred_flag_equals("level", "high"),
                 stale_event_time=0.05, timeout=0.2)
    assert res is None, "Should return None when stale window doesn't cover event"


def test_two_events_and_subscriber_AND():
    e1 = Event(flags=[EventFlag(id="lvl", data_type=str)])
    e2 = Event()
    sub = Subscriber(events=[(e1, pred_flag_equals("lvl", "x")), e2],
                     type=SubscriberType.AND, once=True, stale_event_time=0.5)

    def setter():
        time.sleep(0.05);
        e1.set(data=1, flags={"lvl": "x"})
        time.sleep(0.05);
        e2.set(data=2)

    threading.Thread(target=setter, daemon=True).start()
    res = sub.wait(timeout=1.0)
    assert res is not None
    assert res.data[e1.uid] == 1 and res.data[e2.uid] == 2


def test_event_on_listener_once():
    e = Event(flags=[EventFlag(id="mode", data_type=str)])
    got = []
    done = threading.Event()

    def cb(data):
        got.append(data)
        done.set()

    listener = e.on(callback=cb, predicate=pred_flag_equals("mode", "auto"),
                    once=True, input_data=True, timeout=1.0)
    time.sleep(0.05)
    e.set(data={"k": "v"}, flags={"mode": "auto"})
    # assert done.wait(0.5) is True
    listener.stop()
    assert got == [{"k": "v"}]


def test_event_on_listener_timeout_stops():
    e = Event()
    done = threading.Event()

    def cb(_): done.set()

    listener = e.on(callback=cb, once=True, timeout=0.1)
    # Don't fire anything; listener should time out and stop its thread
    time.sleep(0.3)
    assert not done.is_set(), "Callback must not have been called"
    # Best-effort: ensure internal thread exited
    listener.stop()


def test_pattern_subscriber_attach_and_fire():
    ps = PatternSubscriber(pattern="room_*")
    hits = []
    ps.on(lambda m: hits.append(m), pass_only_data=False)
    e1 = Event(id="room_101")
    time.sleep(0.02)
    e1.set(data={"t": 21})
    # allow dispatch
    time.sleep(0.05)
    assert hits, "Pattern subscriber should attach and receive events"
    assert hits[-1].data[e1.uid] == {"t": 21}


def test_pattern_subscriber_remove_event_cleans_reverse_index():
    ps = PatternSubscriber(pattern="dev_*")
    hits = []
    ps.on(lambda m: hits.append(m))
    e = Event(id="dev_cam")
    time.sleep(0.02)
    ps.remove_event(e)  # explicit detach
    e.set(data={"x": 1})
    time.sleep(0.05)
    # Should not receive any new match caused by e after removal
    assert not hits or all(e.uid not in h.data for h in hits), "No ghost deliveries after remove_event"

    # Also assert reverse index no longer references ps
    subs = active_event_loop.subscribers_by_event.get(e)
    assert not subs or ps not in subs, "Reverse index must not contain the removed pattern subscriber"


def test_event_definition_container_and_uid():
    @event_definition
    class MyEvents(EventContainer):
        ready: Event = Event(flags=[EventFlag('level', str)])
        moved: Event

    evs = MyEvents("robot")
    assert evs.ready.uid == "robot:ready" and evs.moved.uid == "robot:moved"
    hit = []
    sub = Subscriber(events=[(evs.ready, pred_flag_equals("level", "high")), evs.moved],
                     type=SubscriberType.AND)

    def sender():
        evs.ready.set(data=1, flags={"level": "high"})
        time.sleep(0.02)
        evs.moved.set(data=2)

    threading.Thread(target=sender, daemon=True).start()
    res = sub.wait(timeout=1.0)
    assert res is not None and res.data[evs.ready.uid] == 1 and res.data[evs.moved.uid] == 2


def test_get_data_copy_true_isolation():
    e = Event(copy_data_on_set=True)
    src = {"k": ["a", "b"]}
    e.set(data=src)
    src["k"].append("c")
    stored = e.get_data(copy=False)
    assert stored == {"k": ["a", "b"]}


def test_copy_data_on_set_false_aliasing():
    e = Event(copy_data_on_set=False)
    src = {"k": ["a"]}
    e.set(data=src)
    src["k"].append("b")
    assert e.get_data(copy=False) == {"k": ["a", "b"]}


def test_history_pruning_new():
    e = Event()
    e.max_history_time = 0.01
    e.set(data=1)
    assert len(e.history) >= 1
    e._prune_history(now=time.monotonic() + 10.0)
    assert len(e.history) == 0


def test_listener_rate_limit_drop_semantics():
    e = Event()
    count = 0
    done = threading.Event()

    def cb(_):
        nonlocal count
        count += 1

    # 2 Hz max, fire 10 times in ~0.5s → expect <= 2 or 3 depending on edge timing
    listener = e.on(cb, max_rate=2.0, once=False, input_data=True)
    start = time.monotonic()
    for _ in range(10):
        e.set(time.monotonic() - start)
        time.sleep(0.05)
    time.sleep(0.6)
    listener.stop()
    assert count <= 3, f"Rate limit should drop events; got {count}"


def test_pattern_subscriber_future_event_attach():
    ps = PatternSubscriber(pattern="sensor_*")
    hits = []
    ps.on(lambda m: hits.append(m))
    # Create event AFTER subscriber
    e = Event(id="sensor_1")
    time.sleep(0.02)
    e.set(data=123)
    time.sleep(0.05)
    assert hits and e.uid in hits[-1].data and hits[-1].data[e.uid] == 123


def test_subscriber_wait_timeout_callback():
    fired = []
    sub = Subscriber(events=[Event()], timeout=0.1)
    sub.callbacks.timeout.register(lambda: fired.append(True))
    res = sub.wait()  # no events fired
    assert res is None and fired, "Timeout callback should run and wait() should return None"


def test_event_default_id_is_valid():
    e = Event()  # will use generate_uuid()
    # All chars must be alnum or underscore per _check_id
    assert all(c.isalnum() or c == '_' for c in e.id), f"Generated id must satisfy _check_id, got {e.id!r}"


def run_all_tests():
    tests = [name for name, val in globals().items() if name.startswith("test_") and callable(val)]
    print(f"Running {len(tests)} tests...")
    passed = 0
    for t in tests:
        try:
            globals()[t]()
            print(f"✅ {t}")
            passed += 1
        except AssertionError as ae:
            print(f"❌ {t}: {ae}")
            raise
        except Exception as ex:
            print(f"💥 {t}: Unexpected error: {ex}")
            raise
    print(f"All good: {passed}/{len(tests)} passed.")


# ======================================================================================================================

def example_1():
    event1 = Event(id='event1')
    event2 = Event(id='event2', flags=[EventFlag('flag1', str)])
    active_event_loop.addEvent(event1)
    logger = Logger('Test')

    def subscriber_callback(match, *args, **kwargs):
        logger.info('finished')
        logger.info(match)

    # Subscriber with no callback directly added
    subscriber = Subscriber(events=[event1,
                                    (event2, pred_flag_equals('flag1', 'a'))],
                            type=SubscriberType.AND,
                            execute_callback_in_thread=True,
                            once=False)

    # subscriber.on(subscriber_callback, once=True)

    # subscriber.callbacks.finished.register(lambda *args, **kwargs: print('finished'))

    def fire_both_events():
        logger.info("Set both events")
        event1.set("data1")
        time.sleep(0.0)
        event2.set("data2", flags={'flag1': 'a'})

    setTimeout(fire_both_events, 2)
    time.sleep(3)
    result = subscriber.wait(timeout=1, stale_event_time=1.1)
    logger.info(result)

    while True:
        time.sleep(1)


def example_nested_subscribers():
    logger = Logger('Example Nested Subscribers')
    event1 = Event(id='event1')
    event2 = Event(id='event2', flags=[EventFlag('flag1', str)])
    event3 = Event(id='event3', flags=[EventFlag('flag2', str)])

    def subscriber_callback(match, *args, **kwargs):
        logger.info('finished')
        logger.info(match)

    def send_events():
        logger.info("Set events")
        event1.set("data1")
        # time.sleep(1)
        event2.set("data2", flags={'flag1': 'a'})
        # event3.set("data3")

    setTimeout(send_events, 2)
    time.sleep(3)
    subscriber1 = Subscriber(events=[event1, event2], type=SubscriberType.AND, stale_event_time=5)
    subscriber2 = Subscriber(events=[subscriber1, event3],
                             type=SubscriberType.AND,
                             callback=subscriber_callback,
                             stale_event_time=5)
    while True:
        time.sleep(1)


def example_listener():
    logger = Logger('Example Listener')
    event1 = Event(id='event1')
    event2 = Event(id='event2', flags=[EventFlag('flag1', str)])

    subscriber = Subscriber(events=[event1, event2], type=SubscriberType.AND, once=False)

    def listener_callback(match=None, *args, **kwargs):
        logger.info('finished')
        logger.info(match)

    listener = SubscriberListener(subscriber,
                                  listener_callback,
                                  max_rate=1 / 2,
                                  input_data=True)
    listener.start()

    def fire_events():
        logger.info("Set events")
        event1.set("data1")
        event2.set("data2", flags={'flag1': 'a'})

    setInterval(fire_events, 1)

    setTimeout(listener.stop, 10)

    while True:
        time.sleep(1)


def example_pattern_subscriber():
    logger = Logger('Example Pattern Subscriber')

    def pattern_subscriber_callback(data):
        logger.info(f"Pattern subscriber callback: {data}")

    pattern_subscriber = PatternSubscriber(pattern='event2*', id='ps1')
    pattern_subscriber.on(pattern_subscriber_callback)

    event1 = Event(id='event1')
    event2 = Event(id='event2', flags=[EventFlag('flag1', str)])
    event3 = Event(id='event21', flags=[EventFlag('flag2', str)])

    def fire_events():
        logger.info("Set events")
        # event1.set("data1")
        event2.set("data2", flags={'flag1': 'a'})
        event3.set("data3", flags={'flag2': 'b'})

    setTimeout(fire_events, 2)

    while True:
        time.sleep(1)


def example_pattern_subscriber_nested():
    logger = Logger('Example Pattern Subscriber')

    event1 = Event(id='event1')
    event2 = Event(id='event2', flags=[EventFlag('flag1', str)])
    event3 = Event(id='event21', flags=[EventFlag('flag2', str)])

    def subscriber_callback(data):
        logger.info(f"Subscriber callback: {data}")

    subscriber = Subscriber(events=["event2*", event1], type=SubscriberType.AND)
    subscriber.on(subscriber_callback)

    def fire_events():
        logger.info("Set events")
        event1.set("data1")
        # event2.set("data2", flags={'flag1': 'a'})
        event3.set("data3", flags={'flag2': 'b'})

    setTimeout(fire_events, 2)

    while True:
        time.sleep(1)


def example_decorator():
    logger = Logger('Example Decorator')

    @event_definition
    class Events(EventContainer):
        event1: Event = Event(flags=[EventFlag('level', str)])
        event2: Event = Event(flags=[EventFlag('level', str)])

    events = Events('robot')

    def subscriber_callback(data, *args, **kwargs):
        logger.info(f"Subscriber callback: {data}")

    subscriber = Subscriber(events=[events.event1, "*2"], type=SubscriberType.AND)
    subscriber.on(subscriber_callback)

    def fire_events():
        events.event1.set("data1")
        events.event2.set("data2")

    setTimeout(fire_events, 2)

    while True:
        time.sleep(1)


def example_event_on():
    logger = Logger('Example Event On')
    event1 = Event(id='event1')

    def event_callback(data, *args, **kwargs):
        logger.info(f"Event callback: {data}")

    listener = event1.on(event_callback, once=True)

    def fire_event():
        logger.info("Set event")
        event1.set("data1")

    setInterval(fire_event, 2)

    while True:
        time.sleep(1)


if __name__ == '__main__':
    test_pattern_subscriber_attach_and_fire()
