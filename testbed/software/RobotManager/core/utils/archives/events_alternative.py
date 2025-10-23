from __future__ import annotations

import sys
import threading
import time
import typing
from threading import Lock, Condition
import weakref
import collections
from typing import Any, get_origin


# ======================================================================================================================
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


class ConditionEvent(threading.Condition):
    id: str
    resource: 'SharedResource'
    flag: Any
    _parameters_def: list

    def __init__(self, flags=None, history_size=10, id=None):
        """
        :param flags: Optional list of tuples, e.g. [('param1', str), ('param2', type)]
                      which defines the allowed parameters for the event.
        :param history_size: The maximum number of historical events to store.
        :param id: Optional identifier for the event. If not provided explicitly, the event_handler
                   decorator will set the id to the attribute name.
        """
        super().__init__()
        self.id = id
        self.resource = SharedResource()
        self.flag = None
        self._parameters_def = flags if flags is not None else []
        self._last_set_time = None
        # List of listeners. Each listener is a tuple: (callback_ref, flags, once, input_resource)
        # NOTE: After changes, only one-shot (once=True) listeners are stored here.
        self._listeners = []
        # Deque to store history events as tuples: (timestamp, flags)
        self._event_history = collections.deque(maxlen=history_size)

    def on(self, callback, flags=None, once=False, input_resource=True, timeout=None, max_rate=None,
           finished_callback=None) -> 'EventListener':
        """
        Register a listener.

        - If once=True: store a one-shot listener internally. It will be executed in a short-lived thread
          on the next matching event, then removed.
        - If once=False: create and start an EventListener object (with optional max_rate throttling)
          that listens continuously; return that EventListener to the caller. It is NOT stored in the
          internal _listeners list.

        :param callback: The callback to invoke.
        :param flags: Optional dict of filter conditions.
        :param once: See above.
        :param input_resource: If True, pass the event's resource to the callback; otherwise call with no args.
        :param timeout: Optional timeout passed to the EventListener (for continuous listeners).
        :param max_rate: (Hz) Maximum rate at which the callback may be invoked (for continuous listeners).
                         None means no rate limit.
        :param finished_callback: Optional callback to run after each main callback (for continuous or once).
        :return: If once=False, returns the started EventListener instance. If once=True, returns None.
        """
        # For bound methods, keep a weak reference for one-shot storage
        try:
            if hasattr(callback, '__self__') and callback.__self__ is not None:
                callback_ref = weakref.WeakMethod(callback)
            else:
                callback_ref = callback
        except Exception:
            callback_ref = callback

        if once:
            # Store internally; handled on each .set()
            with self:
                self._listeners.append((callback_ref, flags, True, input_resource))
            return None

        # Continuous listener: build an EventListener with optional throttling and return it.
        listener = EventListener(
            event=self,
            callback=callback,
            flags=flags,
            timeout=timeout,
            once=False,
            finished_callback=finished_callback,
            max_rate=max_rate,
            input_resource=input_resource,
        )
        listener.start()
        return listener

    def set(self, resource=None, flags=None):
        """
        Notify all waiting threads and optionally update the shared resource.
        Also triggers any registered one-shot listener callbacks if their filtering flags match.
        Continuous listeners (created via on(..., once=False)) run in their own thread and
        are not stored nor triggered here.
        """
        with self:
            if self._parameters_def:
                # Validate provided flags if parameters are defined.
                if flags is not None:
                    if not isinstance(flags, dict):
                        raise ValueError("Flag must be provided as a dictionary if provided with parameters: {}".format(
                            self._parameters_def))
                    allowed_keys = {p[0] for p in self._parameters_def}
                    for key, value in flags.items():
                        if key not in allowed_keys:
                            raise ValueError("Unexpected parameter: {}".format(key))
                        for param_name, param_type in self._parameters_def:
                            if key == param_name:
                                # Allow the flag value to be a list or tuple of allowed types.
                                if isinstance(value, (list, tuple)):
                                    if not all(isinstance(v, param_type) for v in value):
                                        raise TypeError(
                                            "Each item for parameter '{}' must be of type {}".format(key, param_type))
                                elif not isinstance(value, param_type):
                                    raise TypeError("Parameter '{}' must be of type {}".format(key, param_type))
            self.flag = flags
            self.resource.set(resource)
            timestamp = time.time()
            self._last_set_time = timestamp
            self._event_history.append((timestamp, self.flag))
            self.notify_all()

            # Process one-shot listeners only.
            to_call = []
            remaining_listeners = []
            for callback_ref, listener_flags, once_flag, input_resource in self._listeners:
                if self._check_flag(listener_flags):
                    to_call.append((callback_ref, input_resource))
                    # one-shot => do not re-append
                else:
                    remaining_listeners.append((callback_ref, listener_flags, once_flag, input_resource))
            self._listeners = remaining_listeners

        # Fire matching one-shot listeners in separate threads to avoid blocking.
        for callback_ref, input_resource in to_call:
            if isinstance(callback_ref, weakref.WeakMethod):
                callback_func = callback_ref()
            else:
                callback_func = callback_ref
            if callback_func is None:
                continue
            threading.Thread(target=self._call_listener, args=(callback_func, input_resource)).start()

    def _call_listener(self, callback, input_resource):
        try:
            if input_resource:
                data = self.get_data()
                callback(data)
            else:
                callback()
        except Exception as e:
            print("Error in listener callback:", e)

    def _check_flag(self, filter_params):
        if not filter_params:
            return True
        if not isinstance(self.flag, dict):
            return False
        for key, condition in filter_params.items():
            if key not in self.flag:
                return False
            actual = self.flag[key]

            # Handle if the actual flag is a list/tuple/set.
            if isinstance(actual, (list, tuple, set)):
                # If listener condition is also a collection, check for any intersection.
                if isinstance(condition, (list, tuple, set)):
                    if not set(actual).intersection(condition):
                        return False
                else:
                    if condition not in actual:
                        return False
            else:
                # actual is not a collection
                if isinstance(condition, (list, tuple, set)):
                    if actual not in condition:
                        return False
                elif callable(condition):
                    if not condition(actual):
                        return False
                else:
                    if actual != condition:
                        return False
        return True

    def _match_flags(self, event_flags, conditions):
        """
        Check if the provided event_flags match the filtering conditions.
        """
        if not conditions:
            return True
        if not isinstance(event_flags, dict):
            return False
        for key, condition in conditions.items():
            if key not in event_flags:
                return False
            actual = event_flags[key]
            if isinstance(condition, (list, tuple, set)):
                if actual not in condition:
                    return False
            elif callable(condition):
                if not condition(actual):
                    return False
            else:
                if actual != condition:
                    return False
        return True

    def wait(self, timeout=None, stale_event_time=None, flags: dict = None, resource_filter=None, **filter_params):
        """
        Wait for the condition to be notified and for the flags to match the given criteria.
        Additionally, if stale_event_time is provided, check if an event was set within that time.
        Additionally, if resource_filter is provided, check if the event's resource satisfies the given condition.
        For example, if the resource is a dict, you can wait until a given key has a specific value.
        If the resource is not shaped as assumed by resource_filter, the check safely fails.

        :param timeout: Optional timeout for the wait.
        :param stale_event_time: Optional duration (in seconds) within which a recent event (with matching flags)
                                 will immediately satisfy the wait.
        :param flags: Optional dict of flags to filter event parameters.
        :param resource_filter: Optional filter to apply to the resource. If a dict, each key-value pair is compared
                                against the resource (if it is a dict). Otherwise, a simple equality check is performed.
        :param filter_params: Additional key-value pairs to filter event parameters.
        :return: True if the condition was met, False if the wait timed out.
        """

        def _check_resource(data, resource_filter):
            try:
                if isinstance(resource_filter, dict) and isinstance(data, dict):
                    for key, value in resource_filter.items():
                        if data.get(key) != value:
                            return False
                    return True
                else:
                    return data == resource_filter
            except Exception:
                return False

        with self:
            # Merge filtering conditions.
            conditions = {}
            if flags:
                conditions.update(flags)
            if filter_params:
                conditions.update(filter_params)

            # Check event history for a recent event that matches the conditions.
            if stale_event_time is not None:
                now = time.time()
                for ts, event_flags in self._event_history:
                    if now - ts <= stale_event_time and self._match_flags(event_flags, conditions):
                        if resource_filter is not None:
                            data = self.get_data()
                            if _check_resource(data, resource_filter):
                                return True
                        else:
                            return True

            end_time = time.time() + timeout if timeout is not None else None

            while True:
                remaining = end_time - time.time() if end_time is not None else None
                if remaining is not None and remaining <= 0:
                    return False
                super().wait(remaining)
                if self._check_flag(conditions):
                    if resource_filter is not None:
                        data = self.get_data()
                        if _check_resource(data, resource_filter):
                            return True
                    else:
                        return True

    def get_data(self):
        with self.resource:
            return self.resource.get()

    def clear_data(self):
        with self:
            self.resource.acquire()
            self.resource.set(None)
            self.resource.release()

    def reset(self):
        with self:
            self.flag = None
            self._last_set_time = None
            self.resource.set(None)


def waitForEvents(events: list, timeout=None, wait_for_all=False):
    """
    Wait for one or more events with corresponding flag conditions.

    :param events: A list of tuples (event, flags), where 'flags' is a dictionary used to filter event parameters.
                   Example: [(event1, {'value1': 2}), (event2, {'value33': 'hello'}), (event3, None)]
    :param timeout: Overall timeout in seconds.
    :param wait_for_all: If False, returns as soon as one event is triggered with matching flags;
                         if True, waits until all events are triggered with matching flags.
    :return: If wait_for_all is False, returns the first event that meets its flags.
             If wait_for_all is True, returns a list of events in the same order as provided.
             Returns None if the timeout expires.
    """
    results = []
    lock = threading.Lock()
    done_event = threading.Event()
    threads = []
    start_time = time.time()

    for i, value in enumerate(events):
        if isinstance(value, ConditionEvent):
            events[i] = (value, None)

    def worker(ev, flags, overall_timeout):
        remaining = overall_timeout - (time.time() - start_time) if overall_timeout is not None else None
        if remaining is not None and remaining <= 0:
            return
        ret = ev.wait(timeout=remaining, flags=flags)
        if ret:
            with lock:
                results.append(ev)
            if not wait_for_all:
                done_event.set()

    # Start a thread for each event.
    for ev, flags in events:
        t = threading.Thread(target=worker, args=(ev, flags, timeout))
        t.daemon = True
        t.start()
        threads.append(t)

    if wait_for_all:
        # Wait for all threads to finish, taking into account the overall timeout.
        for t in threads:
            remaining = timeout - (time.time() - start_time) if timeout is not None else None
            t.join(timeout=remaining)
        if len(results) == len(events):
            return results
        else:
            return None
    else:
        done_event.wait(timeout=timeout)
        if results:
            return results[0]
        else:
            return None


class EventListener:
    def __init__(self, event: Condition, callback, flags=None, timeout=None, once=False,
                 finished_callback=None, max_rate=None, input_resource=True):
        """
        Initializes the EventListener.

        :param event: The event to listen for (must be a ConditionEvent or similar with filtering support).
        :param callback: The callback function to execute when the event is triggered.
                         If input_resource=True and the event is a ConditionEvent, the callback receives the
                         event's shared resource; otherwise it is called without arguments.
        :param flags: Optional dictionary of filter conditions to apply (e.g. {'param1': ['a','b','c']}).
        :param timeout: Optional timeout to wait for each event occurrence.
        :param once: If True, the listener will stop after one execution.
        :param finished_callback: Optional callback to execute after the main callback is done.
        :param max_rate: Optional maximum callback firing rate in Hz. If provided, the listener
                         will not invoke the callback more frequently than this rate.
        :param input_resource: If True, pass the event's resource to the callback (ConditionEvent only).
        """
        assert isinstance(event, (Condition, ConditionEvent))
        self.event = event
        self.flags = flags
        self.timeout = timeout
        self.callback = callback
        self.once = once
        self.finished_callback = finished_callback
        self.input_resource = input_resource

        # Rate limiting
        self.max_rate = max_rate
        self._min_interval = (1.0 / max_rate) if (max_rate and max_rate > 0) else None
        self._last_exec_time = 0.0

        self.kill_event = ConditionEvent()
        self._running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True

    def _listen(self):
        while self._running:
            # Wait for a matching event
            if isinstance(self.event, ConditionEvent):
                if self.flags is not None:
                    result = self.event.wait(timeout=self.timeout, flags=self.flags)
                else:
                    result = self.event.wait(timeout=self.timeout)
            else:
                result = self.event.wait(timeout=self.timeout)

            if result == self.kill_event:
                break

            if result is None and self.once:
                break
            # if not result or self.kill_event.is_set():
            #     if self.once:
            #         break
            #     continue

            # Rate limiting: skip callback if fired too recently
            now = time.time()
            if self._min_interval is not None:
                if (now - self._last_exec_time) < self._min_interval:
                    # Drop this event to honor max_rate
                    if self.once:
                        break
                    continue

            self._last_exec_time = now
            self._execute_callback()

            if self.once:
                break

    def _execute_callback(self):
        try:
            if isinstance(self.event, ConditionEvent) and self.input_resource:
                data = self.event.get_data()
                self.callback(data)
            else:
                # Either not a ConditionEvent or the caller does not want input data
                self.callback()
        finally:
            if self.finished_callback:
                try:
                    self.finished_callback()
                except Exception:
                    pass

    def start(self):
        self.thread.start()

    def stop(self):
        self._running = False
        self.kill_event.set()

        if hasattr(self, 'thread') and self.thread is not None and self.thread.is_alive():
            self.thread.join()


# === EVENT DEFINITION =================================================================================================
def event_definition(cls):
    """
    Decorator to make ConditionEvent fields independent per instance.
    Works with and without `from __future__ import annotations`.
    - If a class attribute already stores a ConditionEvent instance, clone its config.
    - If the annotation indicates a ConditionEvent (including Optional/Union), create a fresh one.
    """
    original_init = getattr(cls, "__init__", None)

    # Try to resolve annotations to real types (not strings)
    try:
        module_globals = sys.modules[cls.__module__].__dict__
        # Include class namespace so forward refs to inner names can resolve
        hints = typing.get_type_hints(cls, globalns=module_globals, localns=dict(vars(cls)))
    except Exception:
        # Fallback to raw annotations (may be strings)
        hints = getattr(cls, "__annotations__", {}) or {}

    def _is_condition_event_type(t):
        # When we couldn't resolve types, t could be a string; handle that case
        if t is ConditionEvent:
            return True
        if isinstance(t, str):
            return t == "ConditionEvent" or t.endswith(".ConditionEvent")
        origin = get_origin(t)
        if origin is typing.Union:
            return any(_is_condition_event_type(arg) for arg in typing.get_args(t))
        # Ignore ClassVar[...] etc.
        if origin is typing.ClassVar:
            return False
        return False

    def new_init(self, *args, **kwargs):
        if original_init:
            original_init(self, *args, **kwargs)

        # Walk resolved annotations (or the fallback mapping)
        for attr_name, anno in (hints.items() if isinstance(hints, dict) else []):
            # If the class defines a default instance, clone its config
            default_event = getattr(cls, attr_name, None)
            if isinstance(default_event, ConditionEvent):
                # Preserve the declared flags; give each instance its own ConditionEvent
                new_event = ConditionEvent(flags=default_event._parameters_def, id=attr_name)
                setattr(self, attr_name, new_event)
                continue

            # Otherwise, if the type annotation says it's a ConditionEvent (possibly Optional/Union)
            if _is_condition_event_type(anno):
                setattr(self, attr_name, ConditionEvent(id=attr_name))

        # Also handle the edge case where we couldn't resolve hints
        # and the class provided a default instance without an annotation.
        for attr_name, value in vars(cls).items():
            if isinstance(value, ConditionEvent):
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, ConditionEvent(flags=value._parameters_def, id=attr_name))

    cls.__init__ = new_init
    return cls


# ======================================================================================================================
# Example usage of ConditionEvent with multiple condition filtering and EventListener with filter support
# ==============================================================================================================
# Additional test for .on() functionality.
def test_on_functionality():
    print("\n--- Starting test of .on() functionality ---")
    # Dictionary to collect test results.
    results = {
        "with_input": [],
        "without_input": [],
        "once": 0,
        "filtered": 0,
    }

    # Create a separate event with a flag definition.
    event_test = ConditionEvent(flags=[("test_flag", str)])

    def cb_with_input(resource):
        print("cb_with_input received resource:", resource)
        results["with_input"].append(resource)

    def cb_without_input():
        print("cb_without_input called.")
        results["without_input"].append("called")

    def cb_once(resource):
        print("cb_once received resource:", resource)
        results["once"] += 1

    def cb_filtered(resource):
        print("cb_filtered received resource:", resource)
        results["filtered"] += 1

    # Register listeners for different functionalities.
    # Will always trigger (no flag filtering) and receive the resource.
    event_test.on(cb_with_input, input_resource=True)
    # Will always trigger and should not receive any resource.
    event_test.on(cb_without_input, input_resource=False)
    # Trigger only once.
    event_test.on(cb_once, once=True, input_resource=True)
    # Only trigger when flag "test_flag" equals "pass".
    event_test.on(cb_filtered, flags={"test_flag": "pass"}, input_resource=True)

    # Trigger an event that does not match the filtering flag for cb_filtered.
    print("Triggering event with test_flag: 'fail'")
    event_test.set(resource="Resource 1", flags={"test_flag": "fail"})
    time.sleep(0.5)
    # Trigger an event that will match the filtering flag.
    print("Triggering event with test_flag: 'pass'")
    event_test.set(resource="Resource 2", flags={"test_flag": "pass"})
    time.sleep(0.5)
    # Trigger another event matching the filtering flag.
    print("Triggering event with test_flag: 'pass' again")
    event_test.set(resource="Resource 3", flags={"test_flag": "pass"})
    time.sleep(0.5)

    print("\nTest on() functionality results:")
    print(results)
    print("--- Finished test of .on() functionality ---\n")


if __name__ == '__main__':
    test_on_functionality()
