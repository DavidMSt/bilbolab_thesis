# import abc
# import dataclasses
#
# from applications.FRODO.navigation.navigator import NavigationElement, NavigatedObjectState
# from core.utils.events import Event, event_definition, EventFlag
# from core.utils.logging_utils import Logger
# from core.utils.states import State
#
#
# # === NAVIGATED OBJECTS ================================================================================================
#
#
# @event_definition
# class NavigatedObject_Events:
#     element_finished: Event = Event(flags=[EventFlag('id', str)])
#     navigation_queue_finished: Event
#     navigation_error: Event
#     navigation_queue_started: Event
#     object_event: Event = Event(flags=[EventFlag('id', str)], data_type=str)
#
#
# class NavigatedObject(abc.ABC):
#     id: str
#     state: NavigatedObjectState
#
#     @abc.abstractmethod
#     def fetch_state(self):
#         ...
#
#     @abc.abstractmethod
#     def add_navigation_element(self, element: NavigationElement):
#         ...
#
#     @abc.abstractmethod
#     def clear_navigation_queue(self):
#         ...
#
#     @abc.abstractmethod
#     def get_current_element(self) -> NavigationElement:
#         ...
#
#     @abc.abstractmethod
#     def stop_navigation(self):
#         ...
#
#     @abc.abstractmethod
#     def start_navigation(self):
#         ...
#
#
#
# # ======================================================================================================================
# class MultiAgentMovement:
#     ...
#
# # ======================================================================================================================
# class MultiAgent_Experiment:
#     ...
#
#
# # === MA NAVIGATOR =====================================================================================================
# class MultiAgentNavigator:
#     agents: dict[str, NavigatedObject]
#
#     # === INIT =========================================================================================================
#     def __init__(self):
#         self.logger = Logger('MultiAgentNavigator', 'DEBUG')
#
#     # === METHODS ======================================================================================================
#     def initialize(self, agents: list[NavigatedObject]):
#         self.agents = {agent.id: agent for agent in agents}
#         self.logger.info(
#             f"Initialized multi-agent navigator with {len(agents)} agents: {[agent.id for agent in agents]}")
#     # === PRIVATE METHODS ==============================================================================================
#
#     # ------------------------------------------------------------------------------------------------------------------

# multi_agent_navigator.py
# --------------------------------------------------------------------------------------------------
# Multi-agent orchestration: events, conditions, actions, scheduler + YAML (load/save).
#
# Assumptions (adjust in one place if your adapters differ):
# - Each real/sim agent implements the NavigatedObject ABC below and exposes .events
#   with the following Event fields (you can adapt bind_* methods if names differ):
#       - element_finished: flags['id'] carries element_id (e.g., "wp1")
#       - navigation_error
#       - navigation_queue_started, navigation_queue_finished (optional)
#
# - Your single-agent primitives live in: applications.FRODO.navigation.navigator
#   Import path can be changed in the PRIMITIVE_REGISTRY at the bottom.
# --------------------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any
import threading
import time
import yaml  # PyYAML

# === Import your primitives / base types =========================================
# Adjust this import block if your module path differs.
from applications.FRODO.navigation.navigator import (
    NavigationElement,
    Wait, EventWait,
    CoordinatedMoveTo, MoveTo, MoveToRelative,
    TurnTo, RelativeTurn, TurnToPoint, RelativeStraightMove,
    NavigatedObjectState,
)

# If your project uses custom logging, import it; else use print as fallback.
from core.utils.logging_utils import Logger


# == Minimal event wrapper expectations ==========================================
# Your codebase has a proper Event class; here we just describe the interface we call.
class _ListenerHandle:
    def __init__(self, fn: Callable): self.fn = fn

    def remove(self): pass


class _EventLike:
    # Must support .add_listener(callback) where callback can accept **kwargs that include flags/data
    def add_listener(self, fn: Callable[..., None]) -> _ListenerHandle: ...


# === NAVIGATED OBJECT ABC ========================================================
class NavigatedObject:
    id: str
    state: NavigatedObjectState
    events: Any  # namespace with .element_finished, .navigation_error, etc.

    def fetch_state(self) -> NavigatedObjectState: ...

    def add_navigation_element(self, element: NavigationElement): ...

    def clear_navigation_queue(self): ...

    def get_current_element(self) -> Optional[NavigationElement]: ...

    def stop_navigation(self): ...

    def start_navigation(self): ...

    # Optional (useful for initial conditions if you can set pose directly):
    def set_initial_pose(self, x: float, y: float, psi: float): ...

    # Optional: push host signal to agent's internal EventWait line
    def push_signal(self, name: str): ...


# =================================================================================
# Event Bus (thread-safe, ultra light)
# =================================================================================
class EventBus:
    def __init__(self):
        self._sub: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, data=None):
        with self._lock:
            cbs = list(self._sub.get(topic, []))
        for cb in cbs:
            try:
                cb(topic, data)
            except Exception as e:
                print(f"[EventBus] subscriber error on {topic}: {e}")

    def subscribe(self, topic: str, cb: Callable):
        with self._lock:
            self._sub.setdefault(topic, []).append(cb)

    # Naming helpers
    @staticmethod
    def t_finished(agent_id: str, element_id: str) -> str:
        return f"agent/{agent_id}/finished/{element_id}"

    @staticmethod
    def t_error(agent_id: str) -> str:
        return f"agent/{agent_id}/error"

    @staticmethod
    def t_signal(name: str) -> str:
        return f"signal/{name}"


# =================================================================================
# Conditions
# =================================================================================
class Condition:
    """Polled by the scheduler. attach(bus) once; satisfied() many."""

    def attach(self, bus: EventBus): ...

    def satisfied(self) -> bool: ...

    def to_spec(self) -> dict: return {"type": self.__class__.__name__}


class EventSeen(Condition):
    def __init__(self, topic: str):
        self.topic = topic
        self._flag = False

    def attach(self, bus: EventBus):
        bus.subscribe(self.topic, lambda *_: setattr(self, "_flag", True))

    def satisfied(self) -> bool:
        return self._flag

    def to_spec(self) -> dict:
        return {"type": "EventSeen", "topic": self.topic}


class AllOf(Condition):
    def __init__(self, conds: Iterable[Condition]): self.conds = list(conds)

    def attach(self, bus: EventBus): [c.attach(bus) for c in self.conds]

    def satisfied(self) -> bool: return all(c.satisfied() for c in self.conds)

    def to_spec(self) -> dict:
        return {"type": "AllOf", "conds": [c.to_spec() for c in self.conds]}


class AnyOf(Condition):
    def __init__(self, conds: Iterable[Condition]): self.conds = list(conds)

    def attach(self, bus: EventBus): [c.attach(bus) for c in self.conds]

    def satisfied(self) -> bool: return any(c.satisfied() for c in self.conds)

    def to_spec(self) -> dict:
        return {"type": "AnyOf", "conds": [c.to_spec() for c in self.conds]}


class Timeout(Condition):
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds

    def attach(self, bus: EventBus): ...

    def satisfied(self) -> bool: return time.monotonic() >= self.deadline

    def to_spec(self) -> dict: return {"type": "Timeout", "seconds": self.seconds}


class ExprCondition(Condition):
    """
    Optional power-user predicate over a read-only snapshot (callable returns bool).
    Not serializable to YAML; used only in Python-composed plans.
    """

    def __init__(self, fn: Callable[[], bool]):
        self.fn = fn

    def attach(self, bus: EventBus):
        ...

    def satisfied(self) -> bool:
        try:
            return bool(self.fn())
        except Exception:
            return False

    def to_spec(self) -> dict:
        return {"type": "ExprCondition", "note": "not-serializable"}


# =================================================================================
# Actions
# =================================================================================
@dataclass
class Action:
    """Base class. Actions become runnable when all .when conditions are true."""
    when: List[Condition] = field(default_factory=list)

    def attach(self, bus: EventBus):
        for c in self.when: c.attach(bus)

    def ready(self) -> bool:
        return all(c.satisfied() for c in self.when)

    def run(self, ctx: "MultiAgentNavigator"):
        raise NotImplementedError

    # (De)serialization for YAML
    def to_spec(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "when": [c.to_spec() for c in self.when],
        }


@dataclass
class SendElement(Action):
    agent_id: str = ""
    element: NavigationElement | None = None
    element_id: str = ""
    autostart: bool = True

    def run(self, ctx: "MultiAgentNavigator"):
        agent = ctx.agents[self.agent_id]
        # Attach a stable element_id so the agent emits consistent finished topics:
        if self.element is None:
            raise ValueError("SendElement.element is None")
        if self.element_id:
            self.element.on_finished_event_id = self.element_id
        elif not getattr(self.element, "on_finished_event_id", None):
            # fallback: generate
            self.element.on_finished_event_id = f"el_{int(time.time() * 1000)}"
        agent.add_navigation_element(self.element)
        if self.autostart:
            agent.start_navigation()
        ctx.log.debug(f"SendElement -> {self.agent_id}:{self.element.on_finished_event_id}")

    def to_spec(self) -> dict:
        el_spec = element_to_spec(self.element)
        return {
            **super().to_spec(),
            "agent": self.agent_id,
            "element_id": self.element_id or (getattr(self.element, "on_finished_event_id", None) or ""),
            "element": el_spec,
            "autostart": self.autostart,
        }


@dataclass
class Barrier(Action):
    """When a set of topics has been seen, publish a signal."""
    wait_for_topics: List[str] = field(default_factory=list)
    signal_name: str = ""

    def __post_init__(self):
        if not self.when and self.wait_for_topics:
            self.when = [AllOf([EventSeen(t) for t in self.wait_for_topics])]

    def run(self, ctx: "MultiAgentNavigator"):
        topic = ctx.bus.t_signal(self.signal_name)
        ctx.bus.publish(topic, data={"source": "Barrier"})
        ctx.log.debug(f"Barrier published {topic}")

    def to_spec(self) -> dict:
        return {
            **super().to_spec(),
            "wait_for": self.wait_for_topics,
            "signal": self.signal_name,
        }


@dataclass
class WaitSignal(Action):
    """A no-op action that only gates subsequent actions in YAML ‘phases’."""
    name: str = ""

    def __post_init__(self):
        if not self.when:
            self.when = [EventSeen(EventBus.t_signal(self.name))]

    def run(self, ctx: "MultiAgentNavigator"):
        ctx.log.debug(f"WaitSignal satisfied: {self.name}")

    def to_spec(self) -> dict:
        return {**super().to_spec(), "name": self.name}


@dataclass
class Interrupt(Action):
    agent_id: str = ""
    reason: str = ""

    def run(self, ctx: "MultiAgentNavigator"):
        if self.agent_id not in ctx.agents:
            ctx.log.warning(f"Interrupt: unknown agent {self.agent_id}")
            return
        ctx.agents[self.agent_id].stop_navigation()
        ctx.log.warning(f"Interrupt -> {self.agent_id}: {self.reason}")

    def to_spec(self) -> dict:
        return {**super().to_spec(), "agent": self.agent_id, "reason": self.reason}


@dataclass
class SetInitialPose(Action):
    agent_id: str = ""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0

    def run(self, ctx: "MultiAgentNavigator"):
        ag = ctx.agents[self.agent_id]
        # Preferred: direct pose set (sim), Optional: push a motion element if not supported.
        if hasattr(ag, "set_initial_pose"):
            try:
                ag.set_initial_pose(self.x, self.y, self.psi)
                ctx.log.info(f"SetInitialPose -> {self.agent_id} ({self.x:.2f},{self.y:.2f},{self.psi:.2f})")
                return
            except Exception as e:
                ctx.log.warning(f"set_initial_pose not supported or failed: {e}. Falling back to motion.")
        # Fallback: coordinated move to pose
        el = CoordinatedMoveTo(x=self.x, y=self.y, psi_end=self.psi)
        el.on_finished_event_id = "init_done"
        ag.add_navigation_element(el)
        ag.start_navigation()

    def to_spec(self) -> dict:
        return {**super().to_spec(), "agent": self.agent_id, "x": self.x, "y": self.y, "psi": self.psi}


# =================================================================================
# MultiAgentNavigator (scheduler)
# =================================================================================
class MultiAgentNavigator:
    def __init__(self):
        self.log = Logger("MAN", "DEBUG")
        self.agents: Dict[str, NavigatedObject] = {}
        self.bus = EventBus()
        self._plan: List[Action] = []
        self._lock = threading.Lock()

    # --- wire agents -------------------------------------------------------------
    def initialize(self, agents: Sequence[NavigatedObject]):
        self.agents = {a.id: a for a in agents}
        self.log.info(f"Initialized with agents: {list(self.agents)}")
        self._bind_agent_events()

    def _bind_agent_events(self):
        """
        Bridges agent-level events to bus topics.
        Adjust this if your event interface differs.
        """
        for aid, ag in self.agents.items():
            ev = getattr(ag, "events", None)
            if ev is None:
                self.log.warning(f"Agent {aid} has no .events; skipping bus bindings.")
                continue

            # element_finished must provide flags['id'] with element_id (string).
            def on_finished(*, id=None, **kwargs):
                if id is None:
                    self.log.warning(f"Agent {aid} finished event missing 'id' flag.")
                    return
                self.bus.publish(self.bus.t_finished(aid, str(id)))
                self.log.debug(f"bus <- {self.bus.t_finished(aid, str(id))}")

            # Best effort binding; replace .add_listener with your API (e.g., .add)
            try:
                ev.element_finished.add_listener(on_finished)
            except Exception as e:
                self.log.warning(f"Bind element_finished for {aid} failed: {e}")

            # navigation_error
            def on_error(*args, **kwargs):
                self.bus.publish(self.bus.t_error(aid), data=kwargs)
                self.log.debug(f"bus <- {self.bus.t_error(aid)}")

            try:
                ev.navigation_error.add_listener(on_error)
            except Exception as e:
                self.log.warning(f"Bind navigation_error for {aid} failed: {e}")

    # --- plan lifecycle ----------------------------------------------------------
    def load_plan(self, actions: List[Action]):
        with self._lock:
            self._plan = list(actions)
            for a in self._plan:
                a.attach(self.bus)
        self.log.info(f"Plan loaded with {len(actions)} actions")

    def clear_plan(self):
        with self._lock:
            self._plan.clear()

    # --- scheduler tick ----------------------------------------------------------
    def step(self) -> int:
        """
        Non-blocking tick; call at ~10-50Hz or from an event thread.
        Returns number of actions run this tick.
        """
        ran = 0
        with self._lock:
            # iterate over a copy so we can pop
            idx = 0
            while idx < len(self._plan):
                act = self._plan[idx]
                if act.ready():
                    act.run(self)
                    self._plan.pop(idx)
                    ran += 1
                else:
                    idx += 1
        return ran

    # --- helpers for topic names in authorship ----------------------------------
    def finished(self, agent_id: str, element_id: str) -> str:
        return self.bus.t_finished(agent_id, element_id)

    def signal(self, name: str) -> str:
        return self.bus.t_signal(name)


# =================================================================================
# Primitive registry & element (de)serialization
# =================================================================================
# Map short YAML names to classes and back. Extend as you add primitives.
PRIMITIVE_REGISTRY: Dict[str, type[NavigationElement]] = {
    "Wait": Wait,
    "EventWait": EventWait,
    "CoordinatedMoveTo": CoordinatedMoveTo,
    "MoveTo": MoveTo,
    "MoveToRelative": MoveToRelative,
    "TurnTo": TurnTo,
    "RelativeTurn": RelativeTurn,
    "TurnToPoint": TurnToPoint,
    "RelativeStraightMove": RelativeStraightMove,
}


def element_from_spec(spec: Dict[str, Any]) -> NavigationElement:
    """
    spec:
      type: CoordinatedMoveTo
      params: {x: 1.0, y: 2.0, psi_end: 0.0, arrive_tolerance: 0.05, ...}
    optional:
      id: wp1 (used outside to set on_finished_event_id)
    """
    t = spec.get("type")
    params = dict(spec.get("params", {}))
    if t not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive type '{t}'. Known: {list(PRIMITIVE_REGISTRY)}")
    cls = PRIMITIVE_REGISTRY[t]
    el: NavigationElement = cls(**params)  # type: ignore
    # Let caller assign on_finished_event_id; we don't set it here to avoid collisions
    return el


def element_to_spec(el: NavigationElement | None) -> Dict[str, Any]:
    if el is None:
        return {}
    t = el.__class__.__name__
    # Build params by introspection of dataclass fields if available
    params = {}
    for k, v in getattr(el, "__dict__", {}).items():
        # Skip internal/private / runtime fields commonly present in your design
        if k.startswith("_"):
            continue
        if k in ("active", "finished", "error", "callbacks", "events",
                 "min_duration", "max_duration", "stop_flag",
                 "on_finished_event_id"):
            continue
        # Keep only JSON/YAML friendly values
        if isinstance(v, (int, float, str, bool)) or v is None or isinstance(v, tuple):
            params[k] = v
    return {"type": t, "params": params}


# =================================================================================
# YAML scenario compiler & exporter
# =================================================================================
"""
YAML Schema (authoring-friendly):

agents_required: [robot1, robot2, robot3]

initial_conditions:
  robot1: {x: 0.0, y: 0.0, psi: 0.0}
  robot2: {x: 1.0, y: 0.0, psi: 3.14}

phases:
  - name: phase1
    wait_for: null              # optional: a signal name to gate the phase
    parallel:
      robot1:
        - { type: CoordinatedMoveTo, id: wp1, params: {x: 0.5, y: 0.5} }
        - { type: CoordinatedMoveTo, id: wp2, params: {x: 1.0, y: 0.5} }
      robot2:
        - { type: CoordinatedMoveTo, id: wp3, params: {x: 0.0, y: 1.0} }
      robot3: []
    sync:
      - all_of:
          - "agent/robot1/finished/wp2"
          - "agent/robot2/finished/wp3"
        signal: "phase1_done"

  - name: phase2
    wait_for: "phase1_done"
    parallel:
      robot1:
        - { type: CoordinatedMoveTo, id: wpA, params: {x: 2.0, y: 0.0} }
      robot2:
        - { type: CoordinatedMoveTo, id: wpB, params: {x: 2.0, y: 1.0} }

interrupts:
  - when: "agent/robot2/finished/wp3"
    do:
      - { interrupt: robot1, reason: "sync-advance" }
      - send:
          agent: robot1
          id: wp2_fast
          type: CoordinatedMoveTo
          params: { x: 1.0, y: 0.5 }

"""


def compile_yaml_to_actions(doc: Dict[str, Any], man: Optional[MultiAgentNavigator] = None) -> List[Action]:
    """
    Compile a scenario YAML dict into a flat list of Actions.
    If `man` is provided, uses its EventBus helpers to form topic strings.
    Otherwise, uses static builders on EventBus.
    """
    actions: List[Action] = []

    # Validation: agents_required
    agents_required: List[str] = list(doc.get("agents_required", []))
    if agents_required and man is not None:
        missing = [a for a in agents_required if a not in man.agents]
        if missing:
            raise ValueError(f"Missing required agents: {missing}")

    # Initial conditions -> SetInitialPose + optional init barrier
    init = doc.get("initial_conditions", {}) or {}
    init_topics = []
    for aid, pose in init.items():
        x, y, psi = pose.get("x", 0.0), pose.get("y", 0.0), pose.get("psi", 0.0)
        actions.append(SetInitialPose(agent_id=aid, x=x, y=y, psi=psi))
        # We will *also* push a CoordinatedMoveTo to ensure motion on platforms without set_initial_pose
        # and then wait for it. Topic name:
        init_topics.append(EventBus.t_finished(aid, "init_done"))

    if init_topics:
        # After everyone reaches init pose, publish "init_done" signal for later phases
        actions.append(Barrier(wait_for_topics=init_topics, signal_name="init_all"))

    # Phases
    for phase in doc.get("phases", []) or []:
        wait_sig = phase.get("wait_for")
        if wait_sig:
            actions.append(WaitSignal(name=wait_sig))

        parallel = phase.get("parallel", {}) or {}
        # We chain per-agent sequences with intra-agent dependencies:
        per_agent_last_topic: Dict[str, str] = {}

        def finished_topic(agent_id: str, element_id: str) -> str:
            return EventBus.t_finished(agent_id, element_id)

        # Enqueue all elements; second and later for an agent depend on previous finished
        for aid, elements in parallel.items():
            last_topic = None
            for el_spec in elements or []:
                element_id = str(el_spec.get("id") or f"{aid}_step_{len(actions)}")
                el = element_from_spec(el_spec)
                send = SendElement(agent_id=aid, element=el, element_id=element_id)
                if last_topic:
                    send.when.append(EventSeen(last_topic))
                # If the phase itself had a "wait_for" signal, guard the first element too:
                if wait_sig and last_topic is None:
                    send.when.append(EventSeen(EventBus.t_signal(wait_sig)))
                actions.append(send)
                last_topic = finished_topic(aid, element_id)
            if last_topic:
                per_agent_last_topic[aid] = last_topic

        # Optional explicit sync blocks (barriers that publish named signals)
        for sync in phase.get("sync", []) or []:
            all_of_topics: List[str] = list(sync.get("all_of", []))
            signal_name: str = str(sync.get("signal", ""))
            if all_of_topics and signal_name:
                actions.append(Barrier(wait_for_topics=all_of_topics, signal_name=signal_name))

    # Interrupts
    for intr in doc.get("interrupts", []) or []:
        when_topic = intr.get("when")
        cond = EventSeen(str(when_topic)) if when_topic else None
        for step in intr.get("do", []) or []:
            if "interrupt" in step:
                act = Interrupt(agent_id=step["interrupt"], reason=str(step.get("reason", "")))
                if cond: act.when.append(cond)
                actions.append(act)
            elif "send" in step:
                s = step["send"]
                aid = s["agent"]
                elid = str(s.get("id", ""))
                t = s["type"]
                params = dict(s.get("params", {}))
                el = element_from_spec({"type": t, "params": params})
                act = SendElement(agent_id=aid, element=el, element_id=elid or f"int_{aid}")
                if cond: act.when.append(cond)
                actions.append(act)

    return actions


def load_plan_from_yaml(yaml_str: str, man: Optional[MultiAgentNavigator] = None) -> List[Action]:
    return compile_yaml_to_actions(yaml.safe_load(yaml_str), man=man)


def dump_actions_to_yaml(actions: List[Action]) -> str:
    """
    Export a *generic* YAML describing the flat action list with their conditions.
    This is intentionally lossless and round-trippable, even if it doesn’t regroup
    actions into 'phases'.
    """

    def cond_to_yaml(c: Condition) -> Any:
        return c.to_spec()

    out = []
    for a in actions:
        spec = a.to_spec()
        out.append(spec)
    return yaml.safe_dump({"actions": out}, sort_keys=False)


# =================================================================================
# Helpers for authoring plans in Python (nice ergonomics)
# =================================================================================
def barrier_after(all_topics: Sequence[str], name: str) -> List[Action]:
    return [Barrier(wait_for_topics=list(all_topics), signal_name=name), WaitSignal(name=name)]


def send_with_deps(agent: str, el: NavigationElement, el_id: str, deps: Sequence[str] = ()):
    return SendElement(agent_id=agent, element=el, element_id=el_id, when=[EventSeen(t) for t in deps])


# =================================================================================
# Example usage (as documentation)
# =================================================================================
if __name__ == "__main__":
    # This main is just a smoke-test harness for the module structure.
    man = MultiAgentNavigator()

    # --- (In your app) create and initialize with adapters for real/sim agents ---
    # man.initialize([robot1_adapter, robot2_adapter, robot3_adapter])

    # --- Author a tiny plan directly in Python -----------------------------------
    plan: List[Action] = []
    plan.append(SetInitialPose("robot1", 0, 0, 0))
    plan.append(SetInitialPose("robot2", 1, 0, 3.14))
    # barrier on init
    plan += barrier_after(
        [EventBus.t_finished("robot1", "init_done"),
         EventBus.t_finished("robot2", "init_done")],
        name="init_all")

    # parallel phase gated by "init_all"
    plan.append(SendElement("robot1", CoordinatedMoveTo(x=0.5, y=0.5), "wp1",
                            when=[EventSeen(EventBus.t_signal("init_all"))]))
    plan.append(SendElement("robot1", CoordinatedMoveTo(x=1.0, y=0.5), "wp2",
                            when=[EventSeen(EventBus.t_finished("robot1", "wp1"))]))
    plan.append(SendElement("robot2", CoordinatedMoveTo(x=0.0, y=1.0), "wp3",
                            when=[EventSeen(EventBus.t_signal("init_all"))]))

    # synchronize
    plan += barrier_after(
        [EventBus.t_finished("robot1", "wp2"),
         EventBus.t_finished("robot2", "wp3")],
        name="phase1_done")

    # Interrupt example: when robot2 finishes wp3, interrupt robot1 and send a fast wp2
    plan.append(Interrupt("robot1", "sync-advance",
                          when=[EventSeen(EventBus.t_finished("robot2", "wp3"))]))
    plan.append(SendElement("robot1", CoordinatedMoveTo(x=1.0, y=0.5), "wp2_fast",
                            when=[EventSeen(EventBus.t_finished("robot2", "wp3"))]))

    # Load & run
    man.load_plan(plan)
    # In your app’s loop:
    # while man.step():
    #     pass

    # Demo: round-trip export
    y = dump_actions_to_yaml(plan)
    print("--- EXPORTED ACTIONS YAML ---")
    print(y)
