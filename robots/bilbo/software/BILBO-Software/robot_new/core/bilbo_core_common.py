from core.utils.callbacks import callback_definition
from core.utils.events import event_definition


@callback_definition
class BILBO_Core_Common_Callbacks:
    ...


@event_definition
class BILBO_Core_Common_Events:
    ...


class BILBO_Core_Common:
    callbacks: BILBO_Core_Common_Callbacks
    events: BILBO_Core_Common_Events

    # === INIT =========================================================================================================
    def __init__(self):
        self.callbacks = BILBO_Core_Common_Callbacks()
        self.events = BILBO_Core_Common_Events()
    # === METHODS ======================================================================================================

    # === PRIVATE METHODS ==============================================================================================
