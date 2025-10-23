from __future__ import annotations

import sys
import threading
import time
import typing
import uuid
from collections import deque
from threading import Condition, Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, get_origin

# ======================================================================================================================
# Utilities
# ======================================================================================================================

Predicate = Callable[[Dict[str, Any], Any], bool]  # (flags, resource) -> bool


def pred_in(key, values):
    return lambda f, r: f.get(key) in values


def pred_gt(key, threshold):
    return lambda f, r: f.get(key, 0) > threshold


def pred_resource_key_equals(key, expected):
    return lambda f, r: isinstance(r, dict) and r.get(key) == expected


def pred_flag_key_equals(key, expected):
    return lambda f, r: isinstance(f, dict) and f.get(key) == expected


def flag_contains(flag_key: str, match_value: Any) -> Predicate:
    """
    Returns a predicate that checks if `match_value` is present in the
    flag list (or equals the single flag value) for `flag_key`.

    Works whether the flag value is a list/tuple/set or a single value.
    """

    def _pred(flags, resource):
        if flag_key not in flags:
            return False
        val = flags[flag_key]
        if isinstance(val, (list, tuple, set)):
            return match_value in val
        return val == match_value

    return _pred


class SharedResource:
    def __init__(self, resource=None):
        self.lock = Lock()
        self.resource = resource

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def set(self, value):
        with self.lock:
            self.resource = value

    def get(self):
        return self.resource

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()


# ======================================================================================================================
# Cancelable Wait Handle
# ======================================================================================================================

class WaitHandle:
    """
    Cancelable wait handle returned by ConditionEvent.wait_start().
    Use .wait(timeout) to block; use .cancel() from another thread to abort.
    """
    __slots__ = ("_cv", "_done", "_cancelled", "_result", "_id", "_unregister", "_on_complete")

    def __init__(self, unregister_cb: Callable[[str], None],
                 on_complete: Optional[Callable[['WaitHandle'], None]] = None):
        self._cv = threading.Condition()
        self._done = False
        self._cancelled = False
        self._result = None  # you decide what to store; here: (resource, flags, event)
        self._id = uuid.uuid4().hex
        self._unregister = unregister_cb
        self._on_complete = on_complete

    @property
    def id(self) -> str:
        return self._id

    def wait(self, timeout: Optional[float] = None):
        """Block until completed or cancelled. Returns the result on success; None on timeout/cancel."""
        with self._cv:
            ok = self._cv.wait_for(lambda: self._done or self._cancelled, timeout=timeout)
            if not ok or self._cancelled:
                return None
            return self._result

    def cancel(self):
        """Cancel this waiter (idempotent)."""
        with self._cv:
            if self._done or self._cancelled:
                return
            self._cancelled = True
            self._cv.notify_all()
        self._unregister(self._id)

    # Internal: called by the event when a matching set() occurs
    def _complete(self, result):
        first_time = False
        with self._cv:
            if self._done or self._cancelled:
                return
            self._done = True
            self._result = result
            first_time = True
            self._cv.notify_all()
        self._unregister(self._id)
        if first_time and self._on_complete:
            try:
                self._on_complete(self)
            except Exception:
                pass


# ======================================================================================================================
# ConditionEvent with filtering & cancelable waits
# ======================================================================================================================

class ConditionEvent(Condition):
    """
    A condition-like event that supports:
    - set(resource, flags)
    - filtered waits via flags/resource predicates
    - cancelable waits via WaitHandle
    - one-shot stored listeners (via .on(..., once=True))
    Also keeps a small history for "stale within N seconds" checks.
    """
    id: str
    resource: SharedResource
    flag: Any

    def __init__(self, flags: Optional[List[Tuple[str, type]]] = None, history_size: int = 10,
                 event_id: Optional[str] = None):
        super().__init__()
        self.id = event_id or f"ConditionEvent-{id(self)}"
        self.resource = SharedResource()
        self.flag: Optional[Dict[str, Any]] = None
        self._parameters_def = flags if flags is not None else []
        self._event_history: deque[Tuple[float, Optional[Dict[str, Any]]]] = deque(maxlen=history_size)

        # one-shot listeners stored internally: (weak_or_fn, flags_or_pred, input_resource)
        self._listeners: List[Tuple[Any, Any, bool]] = []

        # active waiters (id -> (WaitHandle, predicate))
        self._waiters: Dict[str, Tuple[WaitHandle, Optional[Predicate]]] = {}

    # ------------------------------- core API --------------------------------

    def set(self, resource: Any = None, flags: Optional[Dict[str, Any]] = None):
        """
        Set the event: update resource/flags, notify waiters/listeners that match.
        Fires one-shot listeners in lightweight threads (or adapt to a shared worker if desired).
        """
        # validate flags against parameters_def (optional)
        if self._parameters_def and flags is not None:
            if not isinstance(flags, dict):
                raise ValueError("flags must be a dict.")
            allowed = {p[0] for p in self._parameters_def}
            for k, v in flags.items():
                if k not in allowed:
                    raise ValueError(f"Unexpected parameter: {k}")
                for pname, ptype in self._parameters_def:
                    if k == pname and not (
                            isinstance(v, (list, tuple)) and all(isinstance(x, ptype) for x in v) or isinstance(v,
                                                                                                                ptype)):
                        raise TypeError(f"Parameter '{k}' must be of type {ptype}")

        # update state + history
        with self:
            self.flag = flags
            self.resource.set(resource)
            self._event_history.append((time.time(), self.flag))

            # snapshot / extract matching waiters
            to_complete: List[WaitHandle] = []
            for _id, (wh, pred) in list(self._waiters.items()):
                if pred is None or self._check_predicate(pred, flags, resource):
                    to_complete.append(wh)
                    # remove now to avoid races / double-completion
                    self._waiters.pop(_id, None)

            # snapshot matching one-shot listeners
            to_call: List[Tuple[Callable, bool]] = []
            remaining: List[Tuple[Any, Any, bool]] = []
            for cb_ref, cond, input_resource in self._listeners:
                callback = cb_ref() if hasattr(cb_ref, "__call__") and getattr(cb_ref, "__self__",
                                                                               None) is not None and hasattr(cb_ref,
                                                                                                             "__func__") else cb_ref
                # weakref.WeakMethod handling (keep simple): try to resolve
                try:
                    if hasattr(cb_ref, "__call__") and hasattr(cb_ref, "__self__"):
                        callback = cb_ref()
                except Exception:
                    callback = cb_ref
                if callback is None:
                    continue
                if cond is None or self._check_condition(cond, flags):
                    to_call.append((callback, input_resource))
                else:
                    remaining.append((cb_ref, cond, input_resource))
            self._listeners = remaining

        # complete waiters (outside lock)
        for wh in to_complete:
            wh._complete((self.resource.get(), self.flag, self))

        # fire one-shot listeners (outside lock)
        for cb, input_res in to_call:
            threading.Thread(target=self._invoke_callback, args=(cb, input_res), daemon=True).start()

    def get_data(self):
        with self.resource:
            return self.resource.get()

    # ------------------------------- waits --------------------------------

    def wait_start(self, *, predicate: Optional[Predicate] = None,
                   stale_event_time: Optional[float] = None,
                   on_complete: Optional[Callable[[WaitHandle], None]] = None) -> WaitHandle:
        """
        Create a cancelable wait and register it. If a recent matching event exists
        and stale_event_time is set, it completes immediately.
        """

        def unregister(_id: str):
            with self:
                self._waiters.pop(_id, None)

        wh = WaitHandle(unregister, on_complete=on_complete)

        # fast-path: recent matching event in history
        if stale_event_time is not None:
            now = time.time()
            # check from newest to oldest
            for ts, ev_flags in reversed(self._event_history):
                if now - ts > stale_event_time:
                    break
                if predicate is None or self._check_predicate(predicate, ev_flags, self.get_data()):
                    wh._complete((self.get_data(), self.flag, self))
                    return wh

        with self:
            self._waiters[wh.id] = (wh, predicate)
        return wh

    def wait(self, *, predicate: Optional[Predicate] = None, timeout: Optional[float] = None,
             stale_event_time: Optional[float] = None):
        """
        Blocking wait built on wait_start(). Returns (resource, flags, event) on success; None on timeout/cancel.
        """
        h = self.wait_start(predicate=predicate, stale_event_time=stale_event_time)
        try:
            return h.wait(timeout=timeout)
        finally:
            # If it timed out (or caller discards), ensure we unregister
            h.cancel()

    # ------------------------------- listeners --------------------------------

    def on(self, callback: Callable, *, predicate: Optional[Predicate] = None, once: bool = False,
           input_resource: bool = True) -> EventListener | None:
        """
        Register a listener. If once=True, it's stored and fired on the next matching set().
        If once=False, returns an EventListener object you can .stop(), but for backward compatibility
        we return an unsubscribe() callback (cancel listener thread) via EventListener.stop().
        """
        if once:
            # store one-shot listener
            cb_ref = _weak_maybe(callback)
            with self:
                self._listeners.append((cb_ref, predicate, input_resource))
            return None

        # continuous listener: own thread that uses cancelable waits
        listener = EventListener(
            event=self,
            callback=callback,
            predicate=predicate,
            input_resource=input_resource
        )
        listener.start()
        return listener  # return a callable that stops it

    # ------------------------------- helpers --------------------------------

    @staticmethod
    def _invoke_callback(callback: Callable, input_resource: bool):
        try:
            if input_resource:
                callback_argc = getattr(callback, "__code__", None).co_argcount if hasattr(callback, "__code__") else 1
                if callback_argc >= 1:
                    callback_arg = callback_arg = callback  # marker to keep style consistency; replaced below
                # Call with resource if desired; ignore signature gymnastics
            # Always call with (resource) if input_resource else no args
            pass
        except Exception:
            pass  # kept to preserve structure; replaced below

    def _invoke_callback(self, callback: Callable, input_resource: bool):
        try:
            if input_resource:
                callback(self.get_data())
            else:
                callback()
        except Exception:
            pass

    @staticmethod
    def _check_predicate(predicate: Predicate, flags: Optional[Dict[str, Any]], resource: Any) -> bool:
        try:
            return predicate(flags or {}, resource)
        except Exception:
            return False

    def _check_condition(self, condition: Any, event_flags: Optional[Dict[str, Any]]) -> bool:
        """Back-compat: allow dict-based filters or callables."""
        if condition is None:
            return True
        if callable(condition):
            return bool(condition(event_flags or {}))
        if not isinstance(event_flags, dict) or not isinstance(condition, dict):
            return False
        for key, expected in condition.items():
            if key not in event_flags:
                return False
            actual = event_flags[key]
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif callable(expected):
                if not expected(actual):
                    return False
            else:
                if actual != expected:
                    return False
        return True


def _weak_maybe(callback):
    """Return a WeakMethod for bound methods; otherwise the function itself."""
    import weakref
    try:
        if hasattr(callback, '__self__') and callback.__self__ is not None:
            return weakref.WeakMethod(callback)
    except Exception:
        pass
    return callback


# ======================================================================================================================
# EventListener using cancelable waits
# ======================================================================================================================

class EventListener:
    """
    Persistent listener that runs a single background thread.
    It waits via cancelable handles, so stop() cancels the in-flight wait immediately.
    """

    def __init__(self, event: ConditionEvent, callback: Callable,
                 predicate: Optional[Predicate] = None,
                 input_resource: bool = True,
                 max_rate: Optional[float] = None):
        self.event = event
        self.callback = callback
        self.predicate = predicate
        self.input_resource = input_resource
        self.max_rate = max_rate
        self._min_interval = (1.0 / max_rate) if (max_rate and max_rate > 0) else None
        self._last_exec = 0.0

        self._stop_ev = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"EventListener-{event.id}", daemon=True)
        self._current_handle: Optional[WaitHandle] = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_ev.set()
        h = self._current_handle
        if h is not None:
            h.cancel()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop_ev.is_set():
            # register a cancelable wait
            self._current_handle = self.event.wait_start(predicate=self.predicate)
            res = self._current_handle.wait(timeout=0.5)  # small slice for responsiveness to stop()
            if res is None:
                # timed out or cancelled; check stop and loop
                continue

            resource, flags, ev = res
            # rate limiting
            now = time.time()
            if self._min_interval is not None and (now - self._last_exec) < self._min_interval:
                continue
            self._last_exec = now

            try:
                if self.input_resource:
                    self.callback(resource)
                else:
                    self.callback()
            except Exception:
                pass

        # ensure any outstanding handle is cancelled
        h = self._current_handle
        if h is not None:
            h.cancel()


# ======================================================================================================================
# wait_any / wait_all (no helper threads)
# ======================================================================================================================

def wait_any(specs: Iterable[Tuple[ConditionEvent, Optional[Predicate]]],
             timeout: Optional[float] = None) -> Optional[
    Tuple[ConditionEvent, Tuple[Any, Dict[str, Any], ConditionEvent]]]:
    """
    Wait until ANY event in specs fires and passes its predicate.
    Returns (winning_event, result_tuple) where result_tuple = (resource, flags, event).
    Returns None on timeout.
    """
    done = threading.Event()
    winner: Dict[str, Any] = {"event": None, "result": None}
    handles: List[WaitHandle] = []

    def make_cb(ev: ConditionEvent):
        def _cb(_wh: WaitHandle):
            # record first winner, signal done
            if not done.is_set():
                winner["event"] = ev
                winner["result"] = _wh.wait(0)  # already completed; get result immediately
                done.set()

        return _cb

    # register all
    for ev, pred in specs:
        h = ev.wait_start(predicate=pred, on_complete=make_cb(ev))
        handles.append(h)

    # wait for first completion
    finished = done.wait(timeout=timeout)
    # cancel the rest
    for h in handles:
        h.cancel()
    if not finished:
        return None
    return typing.cast(ConditionEvent, winner["event"]), typing.cast(Tuple[Any, Dict[str, Any], ConditionEvent],
                                                                     winner["result"])


def wait_all(specs: Iterable[Tuple[ConditionEvent, Optional[Predicate]]],
             timeout: Optional[float] = None) -> Optional[List[Tuple[Any, Dict[str, Any], ConditionEvent]]]:
    """
    Wait until ALL events in specs fire (and pass predicates).
    Returns list of result_tuples (resource, flags, event) in the same order as specs, or None on timeout.
    """
    specs = list(specs)
    remaining = len(specs)
    done = threading.Event()
    results: List[Optional[Tuple[Any, Dict[str, Any], ConditionEvent]]] = [None] * len(specs)
    handles: List[WaitHandle] = []

    def make_cb(idx: int):
        def _cb(_wh: WaitHandle):
            nonlocal remaining
            if results[idx] is None:
                results[idx] = _wh.wait(0)
                remaining -= 1
                if remaining == 0:
                    done.set()

        return _cb

    for idx, (ev, pred) in enumerate(specs):
        h = ev.wait_start(predicate=pred, on_complete=make_cb(idx))
        handles.append(h)

    finished = done.wait(timeout=timeout)
    for h in handles:
        h.cancel()
    if not finished:
        return None
    # type: ignore
    return typing.cast(List[Tuple[Any, Dict[str, Any], ConditionEvent]], results)


# ======================================================================================================================
# Decorator: per-instance ConditionEvent fields
# ======================================================================================================================

def event_definition(cls):
    """
    Decorator to make ConditionEvent fields independent per instance.
    Works with and without `from __future__ import annotations`.
    """
    original_init = getattr(cls, "__init__", None)

    # Try to resolve annotations to real types (not strings)
    try:
        module_globals = sys.modules[cls.__module__].__dict__
        hints = typing.get_type_hints(cls, globalns=module_globals, localns=dict(vars(cls)))
    except Exception:
        hints = getattr(cls, "__annotations__", {}) or {}

    def _is_condition_event_type(t):
        if t is ConditionEvent:
            return True
        if isinstance(t, str):
            return t == "ConditionEvent" or t.endswith(".ConditionEvent")
        origin = get_origin(t)
        if origin is typing.Union:
            return any(_is_condition_event_type(arg) for arg in typing.get_args(t))
        if origin is typing.ClassVar:
            return False
        return False

    def new_init(self, *args, **kwargs):
        if original_init:
            original_init(self, *args, **kwargs)

        # clone defaults or create fresh per annotation
        for attr_name, anno in (hints.items() if isinstance(hints, dict) else []):
            default_event = getattr(cls, attr_name, None)
            if isinstance(default_event, ConditionEvent):
                setattr(self, attr_name, ConditionEvent(flags=default_event._parameters_def, event_id=attr_name))
                continue
            if _is_condition_event_type(anno):
                setattr(self, attr_name, ConditionEvent(event_id=attr_name))

        # unannotated defaults fallback
        for attr_name, value in vars(cls).items():
            if isinstance(value, ConditionEvent) and not hasattr(self, attr_name):
                setattr(self, attr_name, ConditionEvent(flags=value._parameters_def, event_id=attr_name))

    cls.__init__ = new_init
    return cls


# ======================================================================================================================
# Examples / basic tests
# ======================================================================================================================

def _pred_equals(key: str, expected: Any) -> Predicate:
    return lambda flags, res: flags.get(key) == expected


def example_basic_wait_and_cancel():
    print("\n--- example_basic_wait_and_cancel ---")
    ev = ConditionEvent(flags=[("kind", str)])

    # Start a handle and cancel it from another thread
    handle = ev.wait_start(predicate=_pred_equals("kind", "go"))

    def canceller():
        time.sleep(0.2)
        handle.cancel()
        print("canceller: cancelled wait")

    threading.Thread(target=canceller, daemon=True).start()

    res = handle.wait(timeout=1.0)
    print("wait result (should be None):", res)

    # Now actually emit the matching event
    def emitter():
        time.sleep(0.1)
        ev.set(resource={"value": 123}, flags={"kind": "go"})

    threading.Thread(target=emitter, daemon=True).start()

    res2 = ev.wait(predicate=_pred_equals("kind", "go"), timeout=1.0)
    print("second wait (should be tuple):", res2)


def example_listener_once_and_persistent():
    print("\n--- example_listener_once_and_persistent ---")
    ev = ConditionEvent(flags=[("level", str)])

    # one-shot listener
    ev.on(lambda data: print("once got:", data), predicate=_pred_equals("level", "high"), once=True)

    # persistent listener with stop()
    got = []

    def cb(data):
        print("listener got:", data)
        got.append(data)

    stop = ev.on(cb, predicate=lambda f, r: f.get("level") in {"high", "low"}, once=False, input_resource=True)

    ev.set(resource="A", flags={"level": "low"})
    ev.set(resource="B", flags={"level": "high"})
    ev.set(resource="C", flags={"level": "med"})  # filtered out

    time.sleep(0.2)
    stop()  # stop persistent listener
    print("listener collected:", got)


def example_wait_any_all():
    print("\n--- example_wait_any_all ---")
    a = ConditionEvent(flags=[("tag", str)], event_id="A")
    b = ConditionEvent(flags=[("tag", str)], event_id="B")

    # wait_any across two events
    def emitters():
        time.sleep(0.1)
        a.set(resource=1, flags={"tag": "x"})
        time.sleep(0.1)
        b.set(resource=2, flags={"tag": "y"})

    threading.Thread(target=emitters, daemon=True).start()

    winner = wait_any([(a, None), (b, _pred_equals("tag", "y"))], timeout=1.0)
    print("wait_any winner:", (winner[0].id if winner else None), "result:", (winner[1] if winner else None))

    # wait_all across both
    # emit again
    threading.Thread(target=emitters, daemon=True).start()
    all_results = wait_all([(a, None), (b, None)], timeout=1.0)
    print("wait_all results:", all_results)


@event_definition
class MyService:
    # Annotated events will be instantiated per instance
    started: ConditionEvent
    progress: ConditionEvent = ConditionEvent(flags=[("pct", int)])

    def do_work(self):
        self.started.set(resource={"ts": time.time()}, flags={"state": "start"})
        for i in range(0, 101, 20):
            self.progress.set(resource={"i": i}, flags={"pct": i})
            time.sleep(0.01)


def example_decorator_usage():
    print("\n--- example_decorator_usage ---")
    s1 = MyService()
    s2 = MyService()

    # subscribe to s1 progress >= 40, once
    s1.progress.on(lambda data: print("s1 first >=40 data:", data),
                   predicate=lambda f, r: f.get("pct", 0) >= 40, once=True)

    # persistent listener for s2 progress
    stop2 = s2.progress.on(lambda data: print("s2 progress:", data), once=False)

    threading.Thread(target=s1.do_work, daemon=True).start()
    threading.Thread(target=s2.do_work, daemon=True).start()
    time.sleep(0.3)
    stop2()


def example_listener():
    ev = ConditionEvent(flags=[("kind", str)])

    listener = EventListener(event=ev, callback=lambda *args, **kwargs: print("Event received"),
                             predicate=pred_flag_key_equals('kind', 'hello'))
    listener.start()

    while True:
        time.sleep(1)
        ev.set(flags={"kind": "hello"}, resource="world")


def example_on():
    ev = ConditionEvent()
    handle = ev.on(lambda *args, **kwargs: print("Event received"), predicate=pred_flag_key_equals('kind', 'hello'))

    while True:
        time.sleep(1)
        ev.set(flags={"kind": "hello2"}, resource="world")


if __name__ == "__main__":
    # example_listener()
    example_on()
    # example_basic_wait_and_cancel()
    # example_listener_once_and_persistent()
    # example_wait_any_all()
    # example_decorator_usage()
