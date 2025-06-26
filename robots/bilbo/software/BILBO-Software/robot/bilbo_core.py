from core.utils.callbacks import callback_definition
from core.utils.events import event_definition, ConditionEvent


def error_handler(severity, message):
    print(
        f"[{severity}] {message}"
    )

@event_definition
class BILBO_Core_Events:
    resume: ConditionEvent
    repeat: ConditionEvent
    abort: ConditionEvent


@callback_definition
class BILBO_Core_Callbacks:
    ...

# ======================================================================================================================
class BILBO_Core:
    events: BILBO_Core_Events

    def __init__(self):
        self.events = BILBO_Core_Events()

    def setResumeEvent(self, data):
        self.events.resume.set(resource=data)

    def setRepeatEvent(self, data):
        self.events.repeat.set(resource=data)

    def setAbortEvent(self, data):
        self.events.abort.set(resource=data)
