import abc
import dataclasses
import enum
import queue
import threading
import time
import math
from typing import Tuple, Optional, Callable

import qmt

from core.utils.callbacks import callback_definition, CallbackContainer
from core.utils.data import clamp
from core.utils.events import event_definition, Event, EventFlag
from core.utils.exit import register_exit_callback
from core.utils.logging_utils import Logger
from core.utils.states import State


@dataclasses.dataclass
class NavigatedObjectState(State):
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 0.0
    psi_dot: float = 0.0


# === TUNABLE CONSTANTS ================================================================================================
class NavigatorConfig:
    """
    Central place for navigation/control constants.

    NOTE: Values below are conservative defaults and are meant as a starting point.
    - MAX_TRACK_SPEED:    Individual track limit (m/s).
    - TRACK_WIDTH:        Distance between left/right tracks (m).
    - CONTROL_TS:         Control loop period (s).
    - LOOKAHEAD:          Carrot lookahead distance (m). Used to soften heading when far.
    - LIN/ANG PI:         Small integrators to fight friction/imperfections.
    - SOFT_V_LIMIT:       Nominal linear speed cap during navigation (m/s).
    """

    MAX_TRACK_SPEED = 0.2
    TRACK_WIDTH = 0.150

    MAX_FORWARD_SPEED = 0.2  # m/s
    MAX_TURN_SPEED = 3.141  # rad/s

    CONTROL_TS = 0.02

    # Carrot-chasing lookahead (increase for straighter paths)
    LOOKAHEAD = 1

    # Linear (distance) PI
    LIN_KP = 0.6
    LIN_KI = 0.3
    LIN_I_LIMIT = 0.1  # clamp on the integrator (m/s contribution)

    # Angular (heading) PI
    ANG_KP = 2
    ANG_KI = 0.3
    ANG_I_LIMIT = 0.05  # clamp on the integrator (rad/s contribution)

    # ANG_KP = 2
    # ANG_KI = 1
    # ANG_I_LIMIT = 0.8  # clamp on the integrator (rad/s contribution)

    # Nominal caps (tighter than the physical limit to preserve headroom)
    SOFT_V_LIMIT = 0.2  # m/s
    SOFT_OMEGA_LIMIT = 4.0  # rad/s

    # Timeouts / defaults
    DEFAULT_MIN_DURATION = 0.0
    DEFAULT_MAX_DURATION = 60.0  # safety cap for a single element (s)


# === UTILS ============================================================================================================
def _v_omega_to_tracks(v: float, omega: float, track_width: float) -> tuple[float, float]:
    """
    Convert body-frame (v, omega) to left/right track speeds.
    Unicycle-to-differential mapping:
        v = (vr + vl)/2
        omega = (vr - vl)/W
      -> vr = v + (W/2)*omega
         vl = v - (W/2)*omega
    """
    halfW = 0.5 * track_width
    vr = v + halfW * omega
    vl = v - halfW * omega
    return vl, vr


def _saturate_tracks(vl: float, vr: float, limit: float) -> tuple[float, float]:
    """
    Uniformly scale (if needed) to ensure |vl|, |vr| <= limit while preserving curvature.
    """
    max_mag = max(abs(vl), abs(vr), 1e-9)
    if max_mag <= limit:
        return vl, vr
    scale = limit / max_mag
    return vl * scale, vr * scale


# Small helper used by primitives
def _pi_update(err: float, i_acc: float, kp: float, ki: float, i_limit: float, dt: float) -> tuple[float, float]:
    i_acc = clamp(i_acc + err * dt * ki, -i_limit, i_limit)
    u = kp * err + i_acc
    return u, i_acc


# ======================================================================================================================
class TimeRef(enum.StrEnum):
    EXPERIMENT = "EXPERIMENT"  # time referenced to experiment start
    PRIMITIVE = "PRIMITIVE"  # time referenced to primitive start
    ABSOLUTE = "ABSOLUTE"  # time referenced to absolute time


@callback_definition
class NavigationElement_Callbacks:
    started: CallbackContainer
    finished: CallbackContainer
    error: CallbackContainer


@event_definition
class NavigationElement_Events:
    started: Event
    finished: Event = Event(flags=EventFlag('finished', str))
    error: Event


# === PRIMITIVE CONTEXT (what Navigator passes into each primitive step) ===============================================
@dataclasses.dataclass
class PrimitiveContext:
    config: NavigatorConfig
    control_ts: float
    now: float
    t0: float
    exp_t0: float
    internal_event: Event  # NavigatorInternal_Events.event

    def event_matched(self, name: str, stale_window: float = 0.5) -> bool:
        """Return True if the string event occurred within the stale_window."""
        return self.internal_event.has_match_in_window(lambda _f, d: d == name, window=stale_window)


@dataclasses.dataclass
class NavigationElement:
    """
    Base type for all primitives.

    Lifecycle flags are handled by Navigator.
    Each primitive may implement:
      - on_start(state, ctx): optional initialization when primitive becomes active.
      - step(state, ctx) -> (v_cmd, w_cmd, done): compute command and completion.

    The Navigator converts (v_cmd, w_cmd) to track speeds and applies saturation.
    """
    active: bool = False
    finished: bool = False
    error: bool = False

    callbacks: NavigationElement_Callbacks = dataclasses.field(default_factory=NavigationElement_Callbacks)
    events: NavigationElement_Events = dataclasses.field(default_factory=NavigationElement_Events)

    on_finished_event_id: str = ''

    min_duration: float | None = None
    max_duration: float | None = None

    stop_flag: bool = False

    # Navigator-managed timestamps
    _t0: float | None = None
    _exp_t0: float | None = None

    # --- API for subclasses -------------------------------------------------------------------------------------------
    def on_start(self, state, ctx: PrimitiveContext):
        """Optional hook run once when the element is activated."""
        return

    def step(self, state, ctx: PrimitiveContext) -> Tuple[float, float, bool]:
        """Return (v_cmd [m/s], w_cmd [rad/s], done: bool)."""
        return 0.0, 0.0, True  # default: do nothing and finish immediately

    # ------------------------------------------------------------------------------------------------------------------
    def on_finish(self):
        self.events.finished.set(flags={'finished': self.on_finished_event_id})

    # --- Introspection/info for UI/telemetry --------------------------------------------------------------------------
    def _elapsed(self) -> float | None:
        if self._t0 is None:
            return None
        return max(0.0, time.monotonic() - self._t0)

    def getInfo(self) -> dict:
        """
        Return a lightweight info dict that’s safe to serialize/log.
        Subclasses should extend this via super().getInfo() and add their own fields.
        """
        return {
            "name": self.__class__.__name__,
            "type": self.__class__.__name__,
            "active": self.active,
            "finished": self.finished,
            "error": self.error,
            "stop_flag": self.stop_flag,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "elapsed": self._elapsed(),
        }


# === PRIMITIVES =======================================================================================================
# --- WAITING PRIMITIVES -----------------------------------------------------------------------------------------------
class Wait(NavigationElement):
    def getInfo(self) -> dict:
        info = super().getInfo()
        # Generic wait has no extra params
        return info


@dataclasses.dataclass
class TimeWait(Wait):
    duration: float = 0.0
    reference: TimeRef = TimeRef.PRIMITIVE

    def __post_init__(self):
        if self.reference == TimeRef.ABSOLUTE:
            raise ValueError("Absolute time reference is not supported here. Use AbsoluteTimeWait instead.")

    def step(self, state, ctx: PrimitiveContext):
        ref_t0 = ctx.t0 if self.reference == TimeRef.PRIMITIVE else ctx.exp_t0
        done = (time.monotonic() - ref_t0) >= self.duration
        return 0.0, 0.0, done

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "duration": self.duration,
            "reference": str(self.reference),
        })
        return info


@dataclasses.dataclass
class AbsoluteTimeWait(Wait):
    unix_time: float = 0.0
    reference: TimeRef = TimeRef.ABSOLUTE

    def step(self, state, ctx: PrimitiveContext):
        done = time.time() >= self.unix_time
        return 0.0, 0.0, done

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "unix_time": self.unix_time,
            "reference": str(self.reference),
            "time_remaining": max(0.0, self.unix_time - time.time()),
        })
        return info


@dataclasses.dataclass
class EventWait(Wait):
    event: str = ""

    def step(self, state, ctx: PrimitiveContext):
        done = ctx.event_matched(self.event, stale_window=0.5)
        return 0.0, 0.0, done

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "event": self.event,
        })
        return info


# --- MOVEMENT PRIMITIVES ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class _MovementBase(NavigationElement):
    """Common PI and limits for movement primitives."""
    speed: float | None = None
    arrive_tolerance: float = 0.05  # meters or radians (override meaning per primitive)

    # Per-primitive integrators
    _i_lin: float = 0.0
    _i_ang: float = 0.0

    def on_start(self, state, ctx: PrimitiveContext):
        self._i_lin = 0.0
        self._i_ang = 0.0

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "speed": self.speed,
            "arrive_tolerance": self.arrive_tolerance,
        })
        return info


@dataclasses.dataclass
class RelativeStraightMove(_MovementBase):
    """Drive a signed distance along the heading captured at start."""
    distance: float = 0.0
    _goal_x: Optional[float] = None
    _goal_y: Optional[float] = None

    def on_start(self, state, ctx: PrimitiveContext):
        super().on_start(state, ctx)
        psi0 = state.psi
        self._goal_x = state.x + self.distance * math.cos(psi0)
        self._goal_y = state.y + self.distance * math.sin(psi0)

    def step(self, state, ctx: PrimitiveContext):
        assert self._goal_x is not None and self._goal_y is not None
        xg, yg = self._goal_x, self._goal_y

        dx, dy = xg - state.x, yg - state.y
        dist = math.hypot(dx, dy)

        if dist <= self.arrive_tolerance:
            return 0.0, 0.0, True

        # Carrot placement
        look = ctx.config.LOOKAHEAD
        step = max(0.0, dist - look) if dist > 1e-6 else 0.0
        cx = xg - (dx / (dist + 1e-9)) * step
        cy = yg - (dy / (dist + 1e-9)) * step

        psi_des = math.atan2(cy - state.y, cx - state.x)
        e_psi = qmt.wrapToPi(psi_des - state.psi)

        # PI channels
        v_sp = self.speed if self.speed is not None else ctx.config.SOFT_V_LIMIT
        v_pi, self._i_lin = _pi_update(dist, self._i_lin,
                                       ctx.config.LIN_KP, ctx.config.LIN_KI, ctx.config.LIN_I_LIMIT,
                                       ctx.control_ts)
        v_cmd = clamp(v_pi, -v_sp, v_sp) * math.cos(e_psi)

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return v_cmd, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "distance": self.distance,
            "goal": (self._goal_x, self._goal_y),
        })
        return info


@dataclasses.dataclass
class MoveTo(_MovementBase):
    x: float = 0.0
    y: float = 0.0

    def step(self, state, ctx: PrimitiveContext):
        dx, dy = self.x - state.x, self.y - state.y
        dist = math.hypot(dx, dy)

        if dist <= self.arrive_tolerance:
            return 0.0, 0.0, True

        look = ctx.config.LOOKAHEAD
        step = max(0.0, dist - look) if dist > 1e-6 else 0.0
        cx = self.x - (dx / (dist + 1e-9)) * step
        cy = self.y - (dy / (dist + 1e-9)) * step

        psi_des = math.atan2(cy - state.y, cx - state.x)
        e_psi = qmt.wrapToPi(psi_des - state.psi)

        v_sp = self.speed if self.speed is not None else ctx.config.SOFT_V_LIMIT
        v_pi, self._i_lin = _pi_update(dist, self._i_lin,
                                       ctx.config.LIN_KP, ctx.config.LIN_KI, ctx.config.LIN_I_LIMIT,
                                       ctx.control_ts)
        v_cmd = clamp(v_pi, -v_sp, v_sp) * math.cos(e_psi)

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return v_cmd, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "target": (self.x, self.y),
        })
        return info


@dataclasses.dataclass
class MoveToRelative(_MovementBase):
    dx: float = 0.0
    dy: float = 0.0
    _goal_x: Optional[float] = None
    _goal_y: Optional[float] = None

    def on_start(self, state, ctx: PrimitiveContext):
        super().on_start(state, ctx)
        self._goal_x = state.x + self.dx
        self._goal_y = state.y + self.dy

    def step(self, state, ctx: PrimitiveContext):
        assert self._goal_x is not None and self._goal_y is not None
        dx, dy = self._goal_x - state.x, self._goal_y - state.y
        dist = math.hypot(dx, dy)

        if dist <= self.arrive_tolerance:
            return 0.0, 0.0, True

        look = ctx.config.LOOKAHEAD
        step = max(0.0, dist - look) if dist > 1e-6 else 0.0
        cx = self._goal_x - (dx / (dist + 1e-9)) * step
        cy = self._goal_y - (dy / (dist + 1e-9)) * step

        psi_des = math.atan2(cy - state.y, cx - state.x)
        e_psi = qmt.wrapToPi(psi_des - state.psi)

        v_sp = self.speed if self.speed is not None else ctx.config.SOFT_V_LIMIT
        v_pi, self._i_lin = _pi_update(dist, self._i_lin,
                                       ctx.config.LIN_KP, ctx.config.LIN_KI, ctx.config.LIN_I_LIMIT,
                                       ctx.control_ts)
        v_cmd = clamp(v_pi, -v_sp, v_sp) * math.cos(e_psi)

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return v_cmd, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "relative": (self.dx, self.dy),
            "goal": (self._goal_x, self._goal_y),
        })
        return info


@dataclasses.dataclass
class TurnTo(_MovementBase):
    psi: float = 0.0
    arrive_tolerance: float = 0.05  # radians

    def step(self, state, ctx: PrimitiveContext):
        e_psi = qmt.wrapToPi(self.psi - state.psi)
        if abs(e_psi) <= self.arrive_tolerance:
            return 0.0, 0.0, True

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return 0.0, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "psi_target": self.psi,
        })
        return info


@dataclasses.dataclass
class RelativeTurn(_MovementBase):
    dpsi: float = 0.0
    arrive_tolerance: float = 0.05  # radians
    _goal_psi: Optional[float] = None

    def on_start(self, state, ctx: PrimitiveContext):
        super().on_start(state, ctx)
        self._goal_psi = qmt.wrapToPi(state.psi + self.dpsi)

    def step(self, state, ctx: PrimitiveContext):
        assert self._goal_psi is not None
        e_psi = qmt.wrapToPi(self._goal_psi - state.psi)
        if abs(e_psi) <= self.arrive_tolerance:
            return 0.0, 0.0, True

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return 0.0, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "dpsi": self.dpsi,
            "psi_goal": self._goal_psi,
        })
        return info


# --- NEW: TURN TO POINT -----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class TurnToPoint(_MovementBase):
    """
    Pivot in place to face the world-frame point (x, y).
    Useful before a MoveTo if you want straighter approaches.
    """
    x: float = 0.0
    y: float = 0.0
    arrive_tolerance: float = 0.02  # radians

    def step(self, state, ctx: PrimitiveContext):
        psi_des = math.atan2(self.y - state.y, self.x - state.x)
        e_psi = qmt.wrapToPi(psi_des - state.psi)
        if abs(e_psi) <= self.arrive_tolerance:
            return 0.0, 0.0, True

        w_pi, self._i_ang = _pi_update(e_psi, self._i_ang,
                                       ctx.config.ANG_KP, ctx.config.ANG_KI, ctx.config.ANG_I_LIMIT,
                                       ctx.control_ts)
        w_cmd = clamp(w_pi, -ctx.config.SOFT_OMEGA_LIMIT, ctx.config.SOFT_OMEGA_LIMIT)
        return 0.0, w_cmd, False

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "target_point": (self.x, self.y),
        })
        return info


@dataclasses.dataclass
class CoordinatedMoveTo(NavigationElement):
    """
    Composite primitive:
      1) TurnToPoint(x, y)
      2) MoveTo(x, y)
      3) (optional) TurnTo(psi_end)

    Tolerances are exposed so you can tune how 'tight' each stage is.
    """
    x: float = 0.0
    y: float = 0.0
    psi_end: float | None = None

    # Per-stage tolerances / speed
    pre_rotate_tolerance: float = 0.06  # rad for the initial TurnToPoint
    arrive_tolerance: float = 0.05  # m for the MoveTo
    final_heading_tolerance: float = 0.02  # rad for the final TurnTo
    speed: float | None = None  # linear speed cap during MoveTo

    # Internal sub-primitives & stage
    _stage: str = dataclasses.field(default="TURN1", init=False)
    _turn1: TurnToPoint | None = dataclasses.field(default=None, init=False)
    _move: MoveTo | None = dataclasses.field(default=None, init=False)
    _turn2: TurnTo | None = dataclasses.field(default=None, init=False)

    def on_start(self, state, ctx: PrimitiveContext):
        # Build sub-primitives with requested tolerances/speed.
        self._turn1 = TurnToPoint(x=self.x, y=self.y,
                                  arrive_tolerance=self.pre_rotate_tolerance)
        self._move = MoveTo(x=self.x, y=self.y,
                            arrive_tolerance=self.arrive_tolerance,
                            speed=self.speed)
        self._turn2 = TurnTo(psi=self.psi_end if self.psi_end is not None else 0.0,
                             arrive_tolerance=self.final_heading_tolerance) if self.psi_end is not None else None

        # Initialize sub-primitives (reset their integrators, etc.)
        self._turn1.on_start(state, ctx)
        self._move.on_start(state, ctx)
        if self._turn2 is not None:
            self._turn2.on_start(state, ctx)

        # Smart skip: if we already face the target within tolerance, start with MOVE.
        psi_des = math.atan2(self.y - state.y, self.x - state.x)
        if abs(qmt.wrapToPi(psi_des - state.psi)) <= self.pre_rotate_tolerance:
            self._stage = "MOVE"
        else:
            self._stage = "TURN1"

        # Also skip final turn if psi_end is effectively aligned at start (rare but cheap)
        if self._turn2 is not None and self._stage == "TURN1":
            if abs(qmt.wrapToPi(self._turn2.psi - state.psi)) <= self.final_heading_tolerance \
                    and math.hypot(self.x - state.x, self.y - state.y) <= self.arrive_tolerance:
                self._stage = "DONE"

    def step(self, state, ctx: PrimitiveContext) -> tuple[float, float, bool]:
        # Stage machine
        if self._stage == "TURN1":
            v, w, done = self._turn1.step(state, ctx)
            if done:
                self._stage = "MOVE"
                # brief settle: stop this tick; next tick MOVE will start
                return 0.0, 0.0, False
            return v, w, False

        if self._stage == "MOVE":
            v, w, done = self._move.step(state, ctx)
            if done:
                if self._turn2 is not None:
                    self._stage = "TURN2"
                    return 0.0, 0.0, False
                else:
                    self._stage = "DONE"
                    return 0.0, 0.0, True
            return v, w, False

        if self._stage == "TURN2":
            assert self._turn2 is not None
            v, w, done = self._turn2.step(state, ctx)
            if done:
                self._stage = "DONE"
                return 0.0, 0.0, True
            return v, w, False

        # DONE
        return 0.0, 0.0, True

    def getInfo(self) -> dict:
        info = super().getInfo()
        info.update({
            "type": self.__class__.__name__,
            "target": (self.x, self.y),
            "psi_end": self.psi_end,
            "pre_rotate_tolerance": self.pre_rotate_tolerance,
            "arrive_tolerance": self.arrive_tolerance,
            "final_heading_tolerance": self.final_heading_tolerance,
            "speed": self.speed,
            "stage": self._stage,
        })
        return info


# ======================================================================================================================
@event_definition
class Navigator_Events:
    element_started: Event = Event(copy_data_on_set=False)
    element_finished: Event = Event(copy_data_on_set=False)
    element_error: Event = Event(copy_data_on_set=False)
    navigation_started: Event
    navigation_paused: Event
    navigation_resumed: Event
    navigation_finished: Event
    navigation_error: Event


@callback_definition
class Navigator_Callbacks:
    element_started: CallbackContainer
    element_finished: CallbackContainer
    element_error: CallbackContainer
    navigation_started: CallbackContainer
    navigation_paused: CallbackContainer
    navigation_resumed: CallbackContainer
    navigation_finished: CallbackContainer
    navigation_error: CallbackContainer


@event_definition
class NavigatorInternal_Events:
    event: Event = Event(data_type=str)
    stop: Event


class NavigatorStatus(enum.StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class NavigatorElementStatus(enum.StrEnum):
    MOVING = "MOVING"
    WAITING = "WAITING"
    WAITING_FOR_EVENT = "WAITING_FOR_EVENT"
    ERROR = "ERROR"
    DONE = "DONE"


@dataclasses.dataclass
class NavigatorSample:
    status: NavigatorStatus
    element_status: NavigatorElementStatus
    current_element: dict | None
    element_queue: list[dict]
    current_element_index: int
    elements_remaining: int


class NavigatorExecutionMode(enum.StrEnum):
    THREAD = "THREAD"
    EXTERNAL = "EXTERNAL"


class NavigatorSpeedControlMode(enum.StrEnum):
    TRACKS = "TRACKS"
    SPEED_CONTROL = "SPEED_CONTROL"


class Navigator:
    """
    Background worker:
      - dequeues elements and calls their `on_start` (once) and `step` (periodic).
      - converts (v, ω) to (left, right) track speeds and sends them via provided callback.
      - handles lifecycle, min/max duration, and logging.

    You pass in:
      - speed_command_function(left, right)
      - state_fetch_function() -> FRODO_DynamicState
    """
    execution_mode: NavigatorExecutionMode

    speed_command_function: Callable[[float, float], None]
    state_fetch_function: Callable[[], NavigatedObjectState]

    movement_queue: queue.Queue
    status: NavigatorStatus = NavigatorStatus.IDLE

    control_ts: float = NavigatorConfig.CONTROL_TS
    config = NavigatorConfig

    active_element: NavigationElement | None = None
    _exit: bool = False

    events: Navigator_Events
    callbacks: Navigator_Callbacks
    _internal_events: NavigatorInternal_Events

    def __init__(self,
                 mode: NavigatorExecutionMode,
                 speed_control_mode: NavigatorSpeedControlMode,
                 speed_command_function: Callable[[float, float], None],
                 state_fetch_function: Callable[[], NavigatedObjectState]):

        self.mode = mode
        self.speed_control_mode = speed_control_mode
        self.speed_command_function = speed_command_function
        self.state_fetch_function = state_fetch_function
        self.movement_queue = queue.Queue()
        self.logger = Logger("NAVIGATOR", "DEBUG")
        self.callbacks = Navigator_Callbacks()
        self.events = Navigator_Events()
        self._internal_events = NavigatorInternal_Events()

        self._thread = threading.Thread(target=self._task, daemon=True)

        # --- NEW: accounting for indices/remaining
        self._elements_enqueued: int = 0
        self._elements_finished: int = 0
        register_exit_callback(self.stop)

    # === PUBLIC API ===================================================================================================
    def start(self):
        if self.mode == NavigatorExecutionMode.EXTERNAL:
            ...
        elif self.mode == NavigatorExecutionMode.THREAD:
            self._thread.start()
        else:
            raise Exception("Execution mode not supported")
        self.logger.info("Navigator started")

    # ------------------------------------------------------------------------------------------------------------------
    def stop(self):
        self._exit = True
        if self._thread.is_alive():
            self._thread.join()
        self.logger.info("Navigator stopped")

    # ------------------------------------------------------------------------------------------------------------------
    def update(self):
        self._control_step()

    # ------------------------------------------------------------------------------------------------------------------
    def startNavigation(self):
        if self.status in (NavigatorStatus.IDLE, NavigatorStatus.PAUSED, NavigatorStatus.STOPPED):
            self.status = NavigatorStatus.RUNNING
            self.events.navigation_started.set()
            self.callbacks.navigation_started.call()
            self.logger.info("Navigation started")

    # ------------------------------------------------------------------------------------------------------------------
    def stopNavigation(self):
        self.status = NavigatorStatus.STOPPED
        if self.active_element:
            self.active_element.stop_flag = True
        self.speed_command_function(0.0, 0.0)
        self.events.navigation_finished.set()
        self.callbacks.navigation_finished.call()
        self.logger.info("Navigation stopped")

    # ------------------------------------------------------------------------------------------------------------------
    def pauseNavigation(self):
        if self.status == NavigatorStatus.RUNNING:
            self.status = NavigatorStatus.PAUSED
            self.speed_command_function(0.0, 0.0)
            self.events.navigation_paused.set()
            self.callbacks.navigation_paused.call()
            self.logger.info("Navigation paused")

    # ------------------------------------------------------------------------------------------------------------------
    def resumeNavigation(self):
        if self.status == NavigatorStatus.PAUSED:
            self.status = NavigatorStatus.RUNNING
            self.events.navigation_resumed.set()
            self.callbacks.navigation_resumed.call()
            self.logger.info("Navigation resumed")

    # ------------------------------------------------------------------------------------------------------------------
    def runElement(self, element: NavigationElement):
        self.active_element = element
        element.active = True
        element.finished = False
        element.error = False
        element.stop_flag = False
        element._t0 = time.monotonic()
        if element._exp_t0 is None:
            element._exp_t0 = element._t0

        # Default durations
        if element.min_duration is None:
            element.min_duration = self.config.DEFAULT_MIN_DURATION
        if element.max_duration is None:
            element.max_duration = self.config.DEFAULT_MAX_DURATION

        st = self.state_fetch_function()
        ctx = self._make_ctx(element)
        # Per-primitive init
        try:
            element.on_start(st, ctx)
        except Exception as e:
            self.logger.error(f"Element on_start failed: {e}")
            element.error = True

        # Notifications
        self.events.element_started.set(data=element)
        element.events.started.set()
        self.callbacks.element_started.call(element=element)
        element.callbacks.started.call()
        self.logger.debug(f"Element started: {element}")

    # ------------------------------------------------------------------------------------------------------------------
    def abort_element(self):
        """
        Interrupt the currently running element and skip to the next one.
        If no next element is queued, we mark navigation as finished.
        """
        if self.active_element:
            # Signal the active element to stop; control loop will finalize it on the next tick.
            self.active_element.stop_flag = True
            self.active_element.active = False
            self.speed_command_function(0.0, 0.0)
            self.logger.warning("Active element aborted")

            # If nothing else is queued, consider the queue finished.
            if self.movement_queue.qsize() == 0:
                # We cannot call _finish_active() here safely (race with control loop),
                # but we can mark navigation finished once the current element exits.
                # Emit a finished signal now to reflect 'no more work'.
                self.status = NavigatorStatus.STOPPED
                self.events.navigation_finished.set()
                self.callbacks.navigation_finished.call()
                self.logger.info("No further elements queued — navigation finished")
        else:
            # Nothing active; if queue empty, we are done.
            if self.movement_queue.qsize() == 0:
                self.status = NavigatorStatus.STOPPED
                self.events.navigation_finished.set()
                self.callbacks.navigation_finished.call()
                self.logger.info("abort_element called with no active element and empty queue — navigation finished")

    # Backward compatibility (old camelCase API)
    def abortElement(self):
        self.abort_element()

    # ------------------------------------------------------------------------------------------------------------------
    def clearQueue(self):
        cleared = 0
        while True:
            try:
                _ = self.movement_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            self.logger.info(f"Cleared {cleared} queued elements")

    # ------------------------------------------------------------------------------------------------------------------
    def addElement(self,
                   element: NavigationElement,
                   force_start: bool = True,
                   force_element: bool = False):
        """
        Enqueue a navigation element.

        - force_start: if True (default), automatically start running navigation.
        - force_element: if True, preempt the current element (if any) and push this one to the front of the queue.
        """
        if force_element:
            # Preempt current element and put new one at the front of the queue.
            if self.active_element:
                self.active_element.stop_flag = True  # control loop will end it shortly
                self.logger.warning("Forcing element to front — current element will be aborted")

            # queue.Queue uses an internal deque; use its mutex to safely mutate.
            with self.movement_queue.mutex:
                self.movement_queue.queue.appendleft(element)
                self.movement_queue.unfinished_tasks += 1  # keep internal counter consistent
            self._elements_enqueued += 1
        else:
            self.movement_queue.put(element)
            self._elements_enqueued += 1

        if force_start:
            self.startNavigation()

    # ------------------------------------------------------------------------------------------------------------------
    def triggerEvent(self, name: str):
        self._internal_events.event.set(data=name)

    # ------------------------------------------------------------------------------------------------------------------
    def getSample(self) -> NavigatorSample:
        element_status = self._infer_element_status()
        current_info = self.active_element.getInfo() if self.active_element else None

        # Index: how many elements have already finished (0-based for current)
        current_index = self._elements_finished
        # Remaining in queue (does not include the active one)
        remaining = self.movement_queue.qsize()

        sample = NavigatorSample(
            status=self.status,
            element_status=element_status,
            current_element=current_info,
            element_queue=[e.getInfo() for e in self.movement_queue.queue],
            current_element_index=current_index,
            elements_remaining=remaining
        )
        return sample

    # === PRIVATE ======================================================================================================
    def _make_ctx(self, element: NavigationElement) -> PrimitiveContext:
        now = time.monotonic()
        return PrimitiveContext(
            config=self.config,  # type: ignore
            control_ts=self.control_ts,
            now=now,
            t0=element._t0 or now,
            exp_t0=element._exp_t0 or now,
            internal_event=self._internal_events.event
        )

    def _task(self):
        last = time.perf_counter()
        while not self._exit:
            now = time.perf_counter()
            dt = now - last
            if dt < self.control_ts:
                time.sleep(self.control_ts - dt)
                now = time.perf_counter()
            last = now
            try:
                self._control_step()
            except Exception as e:
                self.logger.error(f"Navigator control exception: {e}")
                self.speed_command_function(0.0, 0.0)
                self.events.navigation_error.set(data=str(e))
                self.callbacks.navigation_error.call(error=e)

    def _finish_active(self, ok: bool = True):
        if not self.active_element:
            return
        el = self.active_element
        el.active = False
        el.finished = ok
        el.error = not ok
        self.active_element = None

        # --- NEW: count finished
        self._elements_finished += 1

        if ok:
            self.events.element_finished.set(data=el)
            el.events.finished.set()
            self.callbacks.element_finished.call(element=el)
            el.callbacks.finished.call()

            el.on_finish()

            self.logger.debug(f"Element finished: {el}")
        else:
            self.events.element_error.set(data=el)
            el.events.error.set()
            self.callbacks.element_error.call(element=el)
            el.callbacks.error.call()
            self.logger.error(f"Element error/aborted: {el}")

        self.speed_command_function(0.0, 0.0)

    def _infer_element_status(self) -> NavigatorElementStatus:
        if self.active_element is None:
            # No active element — if we're paused, say WAITING, else DONE.
            if self.status == NavigatorStatus.PAUSED:
                return NavigatorElementStatus.WAITING
            return NavigatorElementStatus.DONE
        el = self.active_element
        if el.error:
            return NavigatorElementStatus.ERROR
        if isinstance(el, EventWait):
            return NavigatorElementStatus.WAITING_FOR_EVENT
        if isinstance(el, Wait):
            return NavigatorElementStatus.WAITING
        # Anything else is a motion primitive (including composites)
        return NavigatorElementStatus.MOVING

    def _control_step(self):
        if self.status != NavigatorStatus.RUNNING:
            self.speed_command_function(0.0, 0.0)
            return

        if self.active_element is None:
            try:
                nxt: NavigationElement = self.movement_queue.get_nowait()
            except queue.Empty:
                self.speed_command_function(0.0, 0.0)
                return
            self.runElement(nxt)

        el = self.active_element
        assert el is not None

        if el.stop_flag:
            self._finish_active(ok=False)
            return

        # Timeouts
        t = time.monotonic()
        t0 = el._t0 or t
        elapsed = t - t0
        if el.max_duration is not None and elapsed > el.max_duration:
            self.logger.warning(f"Element timeout (> {el.max_duration:.1f}s)")
            self._finish_active(ok=False)
            return

        # Primitive step
        st = self.state_fetch_function()
        ctx = self._make_ctx(el)
        v_cmd, w_cmd, done = el.step(st, ctx)

        # Enforce min_duration at finish
        if done and elapsed < (el.min_duration or 0.0):
            done = False

        # Convert & saturate
        if self.speed_control_mode == NavigatorSpeedControlMode.SPEED_CONTROL:
            v = clamp(v_cmd, -self.config.MAX_FORWARD_SPEED, self.config.MAX_FORWARD_SPEED)
            omega = clamp(w_cmd, -self.config.MAX_TURN_SPEED, self.config.MAX_TURN_SPEED)
            self.speed_command_function(v, omega)
        elif self.speed_control_mode == NavigatorSpeedControlMode.TRACKS:
            vl, vr = _v_omega_to_tracks(v_cmd, w_cmd, self.config.TRACK_WIDTH)
            vl, vr = _saturate_tracks(vl, vr, self.config.MAX_TRACK_SPEED)
            self.speed_command_function(vl, vr)

        if done:
            self._finish_active(ok=True)
